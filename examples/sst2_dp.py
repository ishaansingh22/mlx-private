"""SST-2 DP-LoRA: short-input utility + privacy benchmark.

Fallback benchmark used when IMDB DP utility remains unstable.
Runs non-DP and DP-mid on short sentence classification with the same
forced-choice Yes/No evaluation and loss-threshold MIA as IMDB.
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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LEN = 96
LOGICAL_BATCH_SIZE = 8
MICROBATCH_SIZE = 2
MAX_TEXT_TOKENS = 64
MASK_MODE = "label"
EPOCHS = 3
LR = 2e-4
TARGET_DELTA = 1e-5
SEED = 42
N_TRAIN = 1200
N_TEST = 800
PROMPT = "Is the sentiment of this sentence positive? Answer Yes or No."

LORA_RANK = 16
LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

SETTINGS = {
    "non_dp": {"noise_multiplier": 0.0, "l2_norm_clip": 1e10},
    "dp_mid": {"noise_multiplier": 0.5, "l2_norm_clip": 1.0},
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def load_sst2():
    cache = os.path.join(CACHE_DIR, "sst2_raw.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    from datasets import load_dataset

    train = load_dataset("glue", "sst2", split="train")
    val = load_dataset("glue", "sst2", split="validation")
    payload = {
        "train": [{"text": row["sentence"], "label": int(row["label"])} for row in train],
        "validation": [{"text": row["sentence"], "label": int(row["label"])} for row in val],
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(payload, f)
    return payload


def balanced_split(items, n_train, n_test, seed):
    rng = np.random.RandomState(seed)
    pos = [x for x in items if x["label"] == 1]
    neg = [x for x in items if x["label"] == 0]
    n_per_class = (n_train + n_test) // 2
    if len(pos) < n_per_class or len(neg) < n_per_class:
        raise ValueError(
            f"Not enough examples for balanced split "
            f"n_train={n_train} n_test={n_test}: pos={len(pos)} neg={len(neg)}"
        )
    rng.shuffle(pos)
    rng.shuffle(neg)
    balanced = pos[:n_per_class] + neg[:n_per_class]
    rng.shuffle(balanced)
    return balanced[:n_train], balanced[n_train : n_train + n_test]


def _decode_tokens(tokenizer, token_ids, fallback_text):
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    return fallback_text if token_ids else ""


def _build_prompt_and_full_ids(tokenizer, text, answer, seq_len, max_text_tokens):
    text_ids = tokenizer.encode(text)[:max_text_tokens]
    while True:
        clipped = _decode_tokens(tokenizer, text_ids, text) if text_ids else ""
        msgs = [
            {"role": "user", "content": f"{PROMPT}\n\n{clipped}"},
            {"role": "assistant", "content": answer},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(full_text)
        if len(full_ids) <= seq_len + 1 or len(text_ids) == 0:
            return prompt_ids, full_ids
        trim = min(16, len(text_ids))
        text_ids = text_ids[:-trim]


def format_and_tokenize(items, tokenizer, seq_len, *, mask_mode, max_text_tokens):
    all_x, all_y = [], []
    dropped = 0
    for item in items:
        answer = "Yes" if item["label"] == 1 else "No"
        prompt_ids, full_ids = _build_prompt_and_full_ids(
            tokenizer,
            item["text"],
            answer,
            seq_len,
            max_text_tokens,
        )
        resp_start = len(prompt_ids)
        orig_len = len(full_ids)
        if resp_start >= orig_len:
            dropped += 1
            continue

        ids = full_ids
        if orig_len < seq_len + 1:
            ids = ids + [0] * (seq_len + 1 - orig_len)
        ids = ids[: seq_len + 1]

        targets = ids[1 : seq_len + 1]
        mask = [0] * seq_len
        first_resp = max(0, resp_start - 1)
        last_resp = min(orig_len - 1, seq_len)
        if mask_mode == "label":
            if first_resp < last_resp:
                mask[first_resp] = 1
        else:
            for j in range(first_resp, last_resp):
                mask[j] = 1
        if not any(mask):
            dropped += 1
            continue

        all_x.append(ids[:seq_len])
        all_y.append([targets, mask])

    return mx.array(all_x), mx.array(all_y), dropped


def build_model(patch_attention=False):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load(MODEL_NAME)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=4,
        config={"rank": LORA_RANK, "scale": 2.0, "dropout": 0.0, "keys": LORA_KEYS},
    )
    mx.eval(model.parameters())
    if patch_attention:
        ensure_attention_backend_for_per_sample_grads(model, mode="auto", warn=False)
    return model, tokenizer


def _masked_loss(per_tok, mask):
    numer = (per_tok * mask).sum()
    denom = mx.maximum(mask.sum(), mx.array(1.0))
    return numer / denom


def per_sample_loss(model, x, y):
    targets = y[0]
    mask = y[1].astype(mx.float32)
    logits = model(x[None, :])
    per_tok = nn.losses.cross_entropy(logits[0], targets, reduction="none")
    return _masked_loss(per_tok, mask)


def evaluate_accuracy(model, tokenizer, items, *, seq_len, max_text_tokens):
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    correct = 0
    for item in items:
        prompt_ids, _ = _build_prompt_and_full_ids(
            tokenizer,
            item["text"],
            "Yes",
            seq_len,
            max_text_tokens,
        )
        ids = mx.array(prompt_ids)[None, :]
        logits = model(ids)
        last = logits[0, -1, :]
        mx.eval(last)
        pred_pos = last[yes_id].item() > last[no_id].item()
        if pred_pos == (item["label"] == 1):
            correct += 1
    return correct / len(items)


def train_setting(
    setting_name,
    train_x,
    train_y,
    *,
    logical_batch_size,
    microbatch_size,
    seed,
    epochs,
):
    cfg = SETTINGS[setting_name]
    sigma = cfg["noise_multiplier"]
    clip = cfg["l2_norm_clip"]
    is_dp = sigma > 0
    N = int(train_x.shape[0])

    mx.random.seed(seed)
    np.random.seed(seed)
    model, tokenizer = build_model(patch_attention=is_dp)

    if is_dp:
        ps_fn = make_private_loss(model, per_sample_loss, configure_attention_backend=False)
        optimizer = DPOptimizer(
            Adam(learning_rate=LR),
            l2_norm_clip=clip,
            noise_multiplier=sigma,
            target_delta=TARGET_DELTA,
            num_samples=N,
            compile=False,
        )
    else:
        def batch_loss(model, x, y):
            targets = y[:, 0, :]
            mask = y[:, 1, :].astype(mx.float32)
            logits = model(x)
            per_tok = nn.losses.cross_entropy(logits, targets, reduction="none")
            return _masked_loss(per_tok, mask)

        loss_and_grad = nn.value_and_grad(model, batch_loss)
        optimizer = Adam(learning_rate=LR)

    steps_per_epoch = max(1, N // logical_batch_size)
    t0 = time.perf_counter()
    final_loss = float("nan")
    for _ in range(epochs):
        perm = np.random.permutation(N)
        epoch_losses = []
        for step in range(steps_per_epoch):
            idx = perm[step * logical_batch_size : (step + 1) * logical_batch_size]
            if len(idx) < logical_batch_size:
                continue
            xb = train_x[mx.array(idx)]
            yb = train_y[mx.array(idx)]
            if is_dp:
                targets = yb[:, 0, :]
                mask = yb[:, 1, :].astype(mx.float32)
                logits = model(xb)
                per_tok = nn.losses.cross_entropy(logits, targets, reduction="none")
                loss = _masked_loss(per_tok, mask)
                mx.eval(loss)
                optimizer.step_microbatched(
                    model, ps_fn, xb, yb, microbatch_size=microbatch_size,
                )
                mx.eval(model.parameters())
            else:
                loss, grads = loss_and_grad(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
            epoch_losses.append(float(loss.item()))
        final_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

    elapsed = time.perf_counter() - t0
    epsilon = float(optimizer.epsilon) if is_dp else None
    return model, tokenizer, {
        "setting": setting_name,
        "epsilon": epsilon,
        "final_loss": round(final_loss, 4),
        "elapsed_s": round(elapsed, 1),
    }


def score_losses(model, all_x, all_y):
    losses = []
    for i in range(all_x.shape[0]):
        targets = all_y[i, 0, :]
        mask = all_y[i, 1, :].astype(mx.float32)
        logits = model(all_x[i : i + 1])
        per_tok = nn.losses.cross_entropy(logits[0], targets, reduction="none")
        loss = _masked_loss(per_tok, mask)
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
    tpr_at_fpr_001 = 0.0
    for i in range(len(fpr_list)):
        if fpr_list[i] <= 0.01:
            tpr_at_fpr_001 = tpr_list[i]

    return {
        "roc_auc": round(float(auc), 4),
        "tpr@fpr=0.01": round(float(tpr_at_fpr_001), 4),
        "member_loss_mean": round(float(np.mean(member_losses)), 4),
        "nonmember_loss_mean": round(float(np.mean(nonmember_losses)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--logical-batch-size", type=int, default=LOGICAL_BATCH_SIZE)
    parser.add_argument("--microbatch-size", type=int, default=MICROBATCH_SIZE)
    parser.add_argument("--max-text-tokens", type=int, default=MAX_TEXT_TOKENS)
    parser.add_argument("--mask-mode", choices=["assistant", "label"], default=MASK_MODE)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--settings", nargs="+", default=["non_dp", "dp_mid"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    raw = load_sst2()
    train_items, test_items = balanced_split(
        raw["train"], args.n_train, args.n_test, args.seed,
    )

    model, tokenizer = build_model()
    train_x, train_y, _ = format_and_tokenize(
        train_items,
        tokenizer,
        args.seq_len,
        mask_mode=args.mask_mode,
        max_text_tokens=args.max_text_tokens,
    )
    test_x, test_y, _ = format_and_tokenize(
        test_items,
        tokenizer,
        args.seq_len,
        mask_mode=args.mask_mode,
        max_text_tokens=args.max_text_tokens,
    )
    mx.eval(train_x, train_y, test_x, test_y)

    baseline = evaluate_accuracy(
        model,
        tokenizer,
        test_items[:200],
        seq_len=args.seq_len,
        max_text_tokens=args.max_text_tokens,
    )
    print(f"SST-2 zero-shot baseline (200): {baseline:.1%}")
    del model
    mx.clear_cache()

    results = []
    for setting in args.settings:
        model, tok, meta = train_setting(
            setting,
            train_x,
            train_y,
            logical_batch_size=args.logical_batch_size,
            microbatch_size=args.microbatch_size,
            seed=args.seed,
            epochs=args.epochs,
        )
        acc = evaluate_accuracy(
            model,
            tok,
            test_items,
            seq_len=args.seq_len,
            max_text_tokens=args.max_text_tokens,
        )
        train_losses = score_losses(model, train_x, train_y)
        test_losses = score_losses(model, test_x, test_y)
        mia = compute_mia_metrics(train_losses, test_losses)
        meta["accuracy"] = round(acc, 4)
        meta.update(mia)
        results.append(meta)
        del model
        mx.clear_cache()

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = args.output or os.path.join(CACHE_DIR, f"sst2_results_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "seq_len": args.seq_len,
                    "logical_batch_size": args.logical_batch_size,
                    "microbatch_size": args.microbatch_size,
                    "max_text_tokens": args.max_text_tokens,
                    "mask_mode": args.mask_mode,
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "settings": args.settings,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
