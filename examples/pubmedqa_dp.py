"""PubMedQA DP-LoRA: privacy validation on real medical text.

Trains Qwen2.5-0.5B LoRA on PubMedQA (~500 examples) under three privacy
settings and runs a loss-threshold membership inference attack (MIA).

NOTE: The fine-tune does not learn the classification task at this LoRA
rank / data scale (accuracy ≈ majority baseline). This experiment validates
the privacy mechanism on real text, not the model's task utility.

Usage:  python3 examples/pubmedqa_dp.py [--seed 42] [--epochs 5]
Runtime: ~25 min on M1 Pro 16GB
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
SEQ_LEN = 192
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4
TARGET_DELTA = 1e-5
SEED = 42

SETTINGS = {
    "non_dp": {"noise_multiplier": 0.0, "l2_norm_clip": 1e10},
    "dp_mid": {"noise_multiplier": 0.5, "l2_norm_clip": 1.0},
    "dp_strong": {"noise_multiplier": 1.5, "l2_norm_clip": 1.0},
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


# ---- data ---------------------------------------------------------------

def load_pubmedqa():
    cache = os.path.join(CACHE_DIR, "pubmedqa_raw.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    import urllib.request
    base = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=qiaojin/PubMedQA&config=pqa_labeled&split=train"
    )
    rows = []
    for offset in range(0, 1100, 100):
        url = f"{base}&offset={offset}&length=100"
        print(f"  downloading offset={offset}...")
        resp = json.loads(urllib.request.urlopen(url, timeout=30).read())
        batch = resp.get("rows", [])
        rows.extend(batch)
        if len(batch) < 100:
            break

    items = []
    for r in rows:
        row = r["row"]
        items.append({
            "question": row["question"],
            "context": " ".join(row["context"]["contexts"]),
            "answer": row["final_decision"],
        })

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(items, f)

    return items


def format_and_tokenize(items, tokenizer, seq_len):
    """Chat-format, tokenize, build response-only loss mask.

    Returns (x, y) where y has shape (N, 2, seq_len):
      y[:, 0, :] = target token IDs
      y[:, 1, :] = loss mask (1 for response tokens, 0 for prompt)
    """
    all_x, all_y = [], []
    for item in items:
        ctx = item["context"][:400]
        msgs = [
            {"role": "system", "content": "Answer yes, no, or maybe."},
            {"role": "user", "content": f"{ctx}\n\nQuestion: {item['question']}"},
            {"role": "assistant", "content": item["answer"]},
        ]
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer.encode(full_text)

        eval_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True,
        )
        resp_start = len(tokenizer.encode(eval_text))

        orig_len = len(ids)
        if orig_len < seq_len + 1:
            ids = ids + [0] * (seq_len + 1 - orig_len)
        ids = ids[: seq_len + 1]

        targets = ids[1 : seq_len + 1]
        mask = [0] * seq_len
        for j in range(max(0, resp_start - 1), min(orig_len - 1, seq_len)):
            mask[j] = 1

        all_x.append(ids[:seq_len])
        all_y.append([targets, mask])

    return mx.array(all_x), mx.array(all_y)


# ---- model ---------------------------------------------------------------

def load_tokenizer():
    from mlx_lm import load
    _, tokenizer = load(MODEL_NAME)
    mx.clear_cache()
    return tokenizer


def build_model(patch_attention=False):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load(MODEL_NAME)
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


def _masked_loss(per_tok, mask):
    numer = (per_tok * mask).sum()
    denom = mx.maximum(mask.sum(), mx.array(1.0))
    return numer / denom


def per_sample_loss(model, x, y):
    """Response-only loss for a single example. y shape: (2, seq_len)."""
    targets = y[0]
    mask = y[1].astype(mx.float32)
    logits = model(x[None, :])
    per_tok = nn.losses.cross_entropy(logits[0], targets, reduction="none")
    return _masked_loss(per_tok, mask)


# ---- training ------------------------------------------------------------

def train_setting(setting_name, train_x, train_y, *, seed=SEED, epochs=EPOCHS):
    cfg = SETTINGS[setting_name]
    sigma = cfg["noise_multiplier"]
    clip = cfg["l2_norm_clip"]
    is_dp = sigma > 0
    N = int(train_x.shape[0])

    mx.random.seed(seed)
    np.random.seed(seed)
    model, tokenizer = build_model(patch_attention=is_dp)

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
            targets = y[:, 0, :]
            mask = y[:, 1, :].astype(mx.float32)
            logits = model(x)
            per_tok = nn.losses.cross_entropy(logits, targets, reduction="none")
            return _masked_loss(per_tok, mask)

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
                targets = yb[:, 0, :]
                mask = yb[:, 1, :].astype(mx.float32)
                logits = model(xb)
                per_tok = nn.losses.cross_entropy(logits, targets, reduction="none")
                loss = _masked_loss(per_tok, mask)
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

            if step % 50 == 0:
                avg = float(np.mean(epoch_losses[-50:]))
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
        targets = all_y[i, 0, :]
        mask = all_y[i, 1, :].astype(mx.float32)
        logits = model(all_x[i : i + 1])
        per_tok = nn.losses.cross_entropy(logits[0], targets, reduction="none")
        loss = _masked_loss(per_tok, mask)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return losses


def compute_mia_auc(member_losses, nonmember_losses):
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
    return round(float(auc), 4)


# ---- main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    epochs = args.epochs
    seed = args.seed

    print("PubMedQA DP-LoRA Privacy Validation", flush=True)
    print("=" * 60, flush=True)

    raw_items = load_pubmedqa()
    print(f"Loaded {len(raw_items)} PubMedQA examples", flush=True)

    tokenizer = load_tokenizer()

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(raw_items))
    mid = len(raw_items) // 2
    train_items = [raw_items[i] for i in sorted(indices[:mid])]
    test_items = [raw_items[i] for i in sorted(indices[mid:])]
    print(f"Split: {len(train_items)} train, {len(test_items)} test", flush=True)

    train_x, train_y = format_and_tokenize(train_items, tokenizer, SEQ_LEN)
    test_x, test_y = format_and_tokenize(test_items, tokenizer, SEQ_LEN)
    mx.eval(train_x, train_y, test_x, test_y)
    print(f"Tokenized: train={train_x.shape}, test={test_x.shape}", flush=True)

    results = []
    for setting in ["non_dp", "dp_mid", "dp_strong"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  {setting}", flush=True)
        print(f"{'=' * 60}", flush=True)

        model, meta = train_setting(
            setting, train_x, train_y, seed=seed, epochs=epochs,
        )

        print("  Scoring MIA losses...", flush=True)
        train_losses = score_losses(model, train_x, train_y)
        test_losses = score_losses(model, test_x, test_y)
        meta["mia_auc"] = compute_mia_auc(train_losses, test_losses)
        meta["member_loss"] = round(float(np.mean(train_losses)), 4)
        meta["nonmember_loss"] = round(float(np.mean(test_losses)), 4)
        print(
            f"  MIA AUC: {meta['mia_auc']}  "
            f"(member={meta['member_loss']:.3f}, nonmember={meta['nonmember_loss']:.3f})",
            flush=True,
        )

        results.append(meta)
        del model
        mx.clear_cache()

    print(f"\n{'=' * 60}", flush=True)
    print("  PubMedQA Results", flush=True)
    print(f"{'=' * 60}", flush=True)
    hdr = (
        f"{'Setting':<12} {'ε':>8} {'MIA AUC':>9}"
        f" {'Mem.Loss':>9} {'NM.Loss':>9} {'Time':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        eps = f"{r['epsilon']:.2f}" if r["epsilon"] else "∞"
        print(
            f"{r['setting']:<12} {eps:>8} {r['mia_auc']:>9.4f}"
            f" {r['member_loss']:>9.4f} {r['nonmember_loss']:>9.4f}"
            f" {r['elapsed_s']:>6.0f}s"
        )

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, "pubmedqa_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
