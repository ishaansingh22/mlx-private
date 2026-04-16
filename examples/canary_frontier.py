"""Canary corpus MIA frontier.

Generates a synthetic canary corpus with unique identifiers, trains three
LoRA adapters (no-DP, DP-mid, DP-strong), and runs a loss-threshold
membership inference attack to produce the privacy frontier.

Usage:  python3 examples/canary_frontier.py [--seed 42] [--epochs 5]
Runtime: ~8 min on M1 Pro 16GB
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import Adam
from mlx_private import DPOptimizer, make_private_loss
from mlx_private._patch import ensure_attention_backend_for_per_sample_grads

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LEN = 64
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
TARGET_DELTA = 1e-5
SEED = 42
N_CANARIES = 400

SETTINGS = {
    "non_dp": {"noise_multiplier": 0.0, "l2_norm_clip": 1e10},
    "dp_mid": {"noise_multiplier": 0.5, "l2_norm_clip": 1.0},
    "dp_strong": {"noise_multiplier": 1.5, "l2_norm_clip": 1.0},
}

TEMPLATES = [
    "The secret code for entry {i} is {code}. Remember it carefully.",
    "Patient {i} was diagnosed with condition {code} on January 15th.",
    "Account number {i} has balance {code} as of last quarter.",
    "Employee {i} rated performance score {code} in their review.",
    "Document {i} contains classified reference {code} on page three.",
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


# ---- data ----------------------------------------------------------------

def generate_canary_corpus(n, seed):
    rng = np.random.RandomState(seed)
    corpus = []
    for i in range(n):
        t = TEMPLATES[i % len(TEMPLATES)]
        code = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), size=8))
        corpus.append(t.format(i=i, code=code))
    return corpus


def tokenize_corpus(tokenizer, texts, seq_len):
    all_x, all_y = [], []
    pad = tokenizer.pad_token_id or 0
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            ids = ids + [pad] * (seq_len + 1 - len(ids))
        ids = ids[: seq_len + 1]
        all_x.append(ids[:seq_len])
        all_y.append(ids[1 : seq_len + 1])
    return mx.array(all_x), mx.array(all_y)


# ---- model ----------------------------------------------------------------

def build_model(model_name=DEFAULT_MODEL, patch_attention=False):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load(model_name)
    model.freeze()
    linear_to_lora_layers(
        model, num_layers=4,
        config={"rank": 8, "scale": 20.0, "dropout": 0.0,
                "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    )
    mx.eval(model.parameters())
    if patch_attention:
        ensure_attention_backend_for_per_sample_grads(
            model, mode="auto", warn=False,
        )
    return model, tokenizer


def per_sample_loss(model, x, y):
    logits = model(x[None, :])
    return nn.losses.cross_entropy(logits[0], y, reduction="mean")


# ---- training ------------------------------------------------------------

def train_adapter(setting_name, train_x, train_y, *, seed=SEED, epochs=EPOCHS, model_name=DEFAULT_MODEL):
    cfg = SETTINGS[setting_name]
    sigma = cfg["noise_multiplier"]
    clip = cfg["l2_norm_clip"]
    is_dp = sigma > 0
    N = int(train_x.shape[0])

    mx.random.seed(seed)
    np.random.seed(seed)
    model, tokenizer = build_model(model_name=model_name, patch_attention=is_dp)

    if is_dp:
        ps_fn = make_private_loss(
            model, per_sample_loss, configure_attention_backend=False,
        )
        optimizer = DPOptimizer(
            Adam(learning_rate=LR), l2_norm_clip=clip,
            noise_multiplier=sigma, target_delta=TARGET_DELTA,
            num_samples=N, compile=False,
        )
    else:
        def batch_loss(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1), reduction="mean",
            )
        loss_and_grad = nn.value_and_grad(model, batch_loss)
        optimizer = Adam(learning_rate=LR)

    steps_per_epoch = max(1, N // BATCH_SIZE)

    t0 = time.perf_counter()
    final_loss = float("nan")
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        epoch_losses = []
        for step in range(steps_per_epoch):
            idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            if len(idx) < BATCH_SIZE:
                continue
            xb = train_x[mx.array(idx)]
            yb = train_y[mx.array(idx)]

            if is_dp:
                logits = model(xb)
                loss = nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    yb.reshape(-1), reduction="mean",
                )
                mx.eval(loss)
                grads = ps_fn(xb, yb)
                mx.eval(grads)
                optimizer.step(model, grads)
                mx.eval(model.parameters())
            else:
                loss, grads = loss_and_grad(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)

            epoch_losses.append(float(loss.item()))

            if step % 25 == 0:
                avg = float(np.mean(epoch_losses[-25:]))
                elapsed = time.perf_counter() - t0
                print(
                    f"    e{epoch+1} step {step}/{steps_per_epoch}"
                    f"  loss={avg:.4f}  [{elapsed:.0f}s]",
                    flush=True,
                )

        final_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"    epoch {epoch + 1}/{epochs}  loss={final_loss:.4f}", flush=True)

    elapsed = time.perf_counter() - t0
    epsilon = float(optimizer.epsilon) if is_dp else None

    return model, {
        "setting": setting_name,
        "epsilon": epsilon,
        "final_loss": round(final_loss, 4),
        "elapsed_s": round(elapsed, 1),
    }


# ---- MIA -----------------------------------------------------------------

def score_losses(model, all_x, all_y):
    losses = []
    for i in range(all_x.shape[0]):
        logits = model(all_x[i : i + 1])
        loss = nn.losses.cross_entropy(logits[0], all_y[i], reduction="mean")
        mx.eval(loss)
        losses.append(float(loss.item()))
    return losses


def compute_mia_metrics(member_losses, nonmember_losses):
    labels = np.array([1] * len(member_losses) + [0] * len(nonmember_losses))
    scores = np.array(member_losses + nonmember_losses)
    neg_scores = -scores

    sorted_idx = np.argsort(neg_scores)[::-1]
    sorted_labels = labels[sorted_idx]
    n_pos, n_neg = int(labels.sum()), len(labels) - int(labels.sum())

    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0
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

    # TPR at FPR = 0.01
    tpr_at_fpr_001 = 0.0
    for i in range(len(fpr_list)):
        if fpr_list[i] <= 0.01:
            tpr_at_fpr_001 = tpr_list[i]

    # Best balanced accuracy
    thresholds = np.unique(scores)
    best_bal = 0.0
    for t in thresholds:
        preds = (scores <= t).astype(int)
        tpr = ((preds == 1) & (labels == 1)).sum() / max(n_pos, 1)
        tnr = ((preds == 0) & (labels == 0)).sum() / max(n_neg, 1)
        best_bal = max(best_bal, (tpr + tnr) / 2)

    return {
        "roc_auc": round(float(auc), 4),
        "balanced_accuracy": round(float(best_bal), 4),
        "tpr@fpr=0.01": round(float(tpr_at_fpr_001), 4),
        "member_loss_mean": round(float(np.mean(member_losses)), 4),
        "nonmember_loss_mean": round(float(np.mean(nonmember_losses)), 4),
    }


# ---- main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-canaries", type=int, default=N_CANARIES)
    args = parser.parse_args()

    model_name = args.model
    epochs = args.epochs
    seed = args.seed
    n_canaries = args.n_canaries

    print(f"Canary Corpus MIA Frontier — {model_name}")
    print("=" * 60)

    corpus = generate_canary_corpus(n_canaries, seed)
    print(f"Generated {len(corpus)} canary examples")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(corpus))
    mid = len(corpus) // 2
    member_ids = sorted(indices[:mid].tolist())
    nonmember_ids = sorted(indices[mid:].tolist())
    print(f"Split: {len(member_ids)} members, {len(nonmember_ids)} nonmembers")

    model, tokenizer = build_model(model_name=model_name)
    all_x, all_y = tokenize_corpus(tokenizer, corpus, SEQ_LEN)
    train_x = all_x[mx.array(member_ids)]
    train_y = all_y[mx.array(member_ids)]
    mx.eval(all_x, all_y, train_x, train_y)
    del model
    mx.clear_cache()

    results = []
    for setting in ["non_dp", "dp_mid", "dp_strong"]:
        print(f"\n{'=' * 60}")
        print(f"  {setting}")
        print(f"{'=' * 60}")

        model, meta = train_adapter(
            setting, train_x, train_y, seed=seed, epochs=epochs,
            model_name=model_name,
        )

        all_losses = score_losses(model, all_x, all_y)
        m_losses = [all_losses[i] for i in member_ids]
        nm_losses = [all_losses[i] for i in nonmember_ids]
        mia = compute_mia_metrics(m_losses, nm_losses)
        meta.update(mia)

        eps_str = f"ε={meta['epsilon']:.2f}" if meta["epsilon"] else "ε=∞"
        print(f"  {eps_str}  AUC={mia['roc_auc']}  bal_acc={mia['balanced_accuracy']}  TPR@1%={mia['tpr@fpr=0.01']}")

        results.append(meta)
        del model
        mx.clear_cache()

    print(f"\n{'=' * 60}")
    print("  Canary MIA Frontier")
    print(f"{'=' * 60}")
    hdr = f"{'Setting':<12} {'ε':>8} {'ROC-AUC':>9} {'Bal.Acc':>9} {'TPR@1%':>8} {'Mem.Loss':>9} {'NM.Loss':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        eps = f"{r['epsilon']:.2f}" if r["epsilon"] else "∞"
        print(
            f"{r['setting']:<12} {eps:>8} {r['roc_auc']:>9.4f}"
            f" {r['balanced_accuracy']:>9.4f}"
            f" {r['tpr@fpr=0.01']:>8.4f}"
            f" {r['member_loss_mean']:>9.4f} {r['nonmember_loss_mean']:>9.4f}"
        )

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, "canary_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
