"""Loss-threshold membership inference attack.

Scores every example in the corpus using a trained adapter, then computes
MIA metrics by thresholding on per-example loss.

Usage:
    python attack.py --run-dir runs/non_dp --split split.json --corpus corpus.jsonl --output results/non_dp.json
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_private._patch import ensure_attention_backend_for_per_sample_grads


def load_model_with_adapter(meta: dict, adapter_path: str):
    from mlx_lm import load as load_model
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load_model(meta["model_name"])
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=meta["lora_layers"],
        config={
            "rank": meta["lora_rank"],
            "scale": 20.0,
            "dropout": 0.0,
            "keys": meta["lora_keys"],
        },
    )

    weights = dict(mx.load(adapter_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model, tokenizer


def tokenize_corpus(tokenizer, corpus: list[str], seq_len: int):
    all_x, all_y = [], []
    for text in corpus:
        ids = tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            ids = ids + [tokenizer.pad_token_id or 0] * (seq_len + 1 - len(ids))
        ids = ids[: seq_len + 1]
        all_x.append(ids[:seq_len])
        all_y.append(ids[1 : seq_len + 1])
    return mx.array(all_x), mx.array(all_y)


def score_losses(
    model, all_x, all_y, response_start: int | None = None,
) -> list[float]:
    """Compute per-example cross-entropy loss.

    Args:
        response_start: If set, only compute loss on tokens at positions
            >= response_start (response-only scoring). None = full sequence.
    """
    losses = []
    for i in range(all_x.shape[0]):
        xi = all_x[i : i + 1]
        yi = all_y[i : i + 1]
        logits = model(xi)
        pred = logits
        target = yi
        if response_start is not None and response_start > 0:
            start = max(0, response_start - 1)
            pred = pred[:, start:, :]
            target = target[:, start:]
        loss = nn.losses.cross_entropy(
            pred.reshape(-1, pred.shape[-1]),
            target.reshape(-1),
            reduction="mean",
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
    return losses


def compute_mia_metrics(member_losses: list[float], nonmember_losses: list[float]):
    """Loss-threshold MIA: lower loss -> predict member.

    Returns metrics dict including ROC-AUC, balanced accuracy,
    TPR@FPR=0.01, TPR@FPR=0.001, and full ROC curve data.
    """
    labels = np.array([1] * len(member_losses) + [0] * len(nonmember_losses))
    scores = np.array(member_losses + nonmember_losses)
    neg_scores = -scores

    sorted_idx = np.argsort(neg_scores)[::-1]
    sorted_labels = labels[sorted_idx]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    tpr_list, fpr_list = [0.0], [0.0]
    tp, fp = 0, 0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / max(n_pos, 1))
        fpr_list.append(fp / max(n_neg, 1))

    auc = sum(
        (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2
        for i in range(len(fpr_list) - 1)
    )

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    tpr_at_fpr = {}
    for target_fpr in [0.01, 0.001]:
        mask = fpr_arr <= target_fpr
        tpr_at_fpr[f"tpr@fpr={target_fpr}"] = round(float(tpr_arr[mask].max()), 4) if mask.any() else 0.0

    thresholds = np.unique(scores)
    best_bal_acc = 0.0
    best_thresh = float(thresholds[0])
    for t in thresholds:
        preds = (scores <= t).astype(int)
        tp_t = ((preds == 1) & (labels == 1)).sum()
        tn_t = ((preds == 0) & (labels == 0)).sum()
        tpr_t = tp_t / max(n_pos, 1)
        tnr_t = tn_t / max(n_neg, 1)
        bal = (tpr_t + tnr_t) / 2
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_thresh = float(t)

    return {
        "roc_auc": round(float(auc), 4),
        "balanced_accuracy": round(float(best_bal_acc), 4),
        **tpr_at_fpr,
        "best_threshold": round(best_thresh, 6),
        "n_members": int(n_pos),
        "n_nonmembers": int(n_neg),
        "member_loss_mean": round(float(np.mean(member_losses)), 4),
        "nonmember_loss_mean": round(float(np.mean(nonmember_losses)), 4),
        "member_loss_std": round(float(np.std(member_losses)), 4),
        "nonmember_loss_std": round(float(np.std(nonmember_losses)), 4),
        "roc_curve": {"fpr": [round(f, 6) for f in fpr_list], "tpr": [round(t, 6) for t in tpr_list]},
    }


def run_attack(
    run_dir: str,
    split_path: str,
    corpus_path: str,
    output_path: str,
    response_start: int | None = None,
):
    """Run loss-threshold MIA and save results. Returns metrics dict."""
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)

    adapter_path = os.path.join(run_dir, "adapter.npz")
    model, tokenizer = load_model_with_adapter(meta, adapter_path)
    ensure_attention_backend_for_per_sample_grads(model, mode="auto", warn=False)

    with open(split_path) as f:
        split = json.load(f)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line)["text"])

    seq_len = meta.get("seq_len", 64)
    all_x, all_y = tokenize_corpus(tokenizer, corpus, seq_len)
    mx.eval(all_x, all_y)

    all_losses = score_losses(model, all_x, all_y, response_start=response_start)

    member_set = set(split["member_ids"])
    nonmember_set = set(split["nonmember_ids"])
    member_losses = [all_losses[i] for i in range(len(corpus)) if i in member_set]
    nonmember_losses = [all_losses[i] for i in range(len(corpus)) if i in nonmember_set]

    metrics = compute_mia_metrics(member_losses, nonmember_losses)
    metrics["setting"] = meta.get("setting", "unknown")
    metrics["epsilon"] = meta.get("epsilon")
    if response_start is not None:
        metrics["response_start"] = response_start

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    raw_path = output_path.replace(".json", "_losses.json")
    with open(raw_path, "w") as f:
        json.dump({
            "all_losses": all_losses,
            "member_ids": split["member_ids"],
            "nonmember_ids": split["nonmember_ids"],
        }, f)

    del model
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response-start", type=int, default=None,
                        help="Token position where response begins (for response-only scoring)")
    args = parser.parse_args()

    metrics = run_attack(
        args.run_dir, args.split, args.corpus, args.output,
        response_start=args.response_start,
    )

    print(f"  ROC-AUC: {metrics['roc_auc']}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']}")
    print(f"  TPR@FPR=0.01: {metrics.get('tpr@fpr=0.01', 'N/A')}")
    print(f"  TPR@FPR=0.001: {metrics.get('tpr@fpr=0.001', 'N/A')}")
    print(f"  Member mean loss: {metrics['member_loss_mean']}")
    print(f"  Nonmember mean loss: {metrics['nonmember_loss_mean']}")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
