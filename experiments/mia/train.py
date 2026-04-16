"""MIA experiment: train LoRA adapters with and without DP.

Usage:
    python train.py --setting non_dp --output runs/non_dp
    python train.py --setting dp_mid  --output runs/dp_mid
    python train.py --setting dp_strong --output runs/dp_strong
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
from mlx.utils import tree_flatten

from mlx_private import DPOptimizer, make_private_loss
from mlx_private._patch import ensure_attention_backend_for_per_sample_grads

SETTINGS = {
    "non_dp": {"noise_multiplier": 0.0, "l2_norm_clip": 1e10},
    "dp_mid": {"noise_multiplier": 0.5, "l2_norm_clip": 1.0},
    "dp_strong": {"noise_multiplier": 1.5, "l2_norm_clip": 1.0},
}

DEFAULT_CFG = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "lora_layers": 4,
    "lora_rank": 8,
    "lora_keys": ["self_attn.q_proj", "self_attn.v_proj"],
    "lr": 1e-4,
    "batch_size": 4,
    "seq_len": 64,
    "epochs": 3,
    "seed": 42,
    "target_delta": 1e-5,
}


def load_split(path: str):
    """Load a split manifest (JSON with 'member_ids' and 'nonmember_ids')."""
    with open(path) as f:
        return json.load(f)


def build_model(cfg: dict):
    from mlx_lm import load as load_model
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load_model(cfg["model_name"])
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=cfg["lora_layers"],
        config={
            "rank": cfg["lora_rank"],
            "scale": 20.0,
            "dropout": 0.0,
            "keys": cfg["lora_keys"],
        },
    )
    mx.eval(model.parameters())
    ensure_attention_backend_for_per_sample_grads(model, mode="auto", warn=False)
    return model, tokenizer


def tokenize_corpus(tokenizer, corpus: list[str], seq_len: int):
    """Tokenize a list of strings, truncate/pad to seq_len. Returns (x, y) for LM."""
    all_x, all_y = [], []
    for text in corpus:
        ids = tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            ids = ids + [tokenizer.pad_token_id or 0] * (seq_len + 1 - len(ids))
        ids = ids[: seq_len + 1]
        all_x.append(ids[:seq_len])
        all_y.append(ids[1 : seq_len + 1])
    return mx.array(all_x), mx.array(all_y)


def per_sample_loss(model, x, y):
    logits = model(x[None, :])
    return nn.losses.cross_entropy(logits[0], y, reduction="mean")


def train(cfg: dict, train_x, train_y, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    mx.random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model, tokenizer = build_model(cfg)

    sigma = cfg["noise_multiplier"]
    clip = cfg["l2_norm_clip"]
    N = int(train_x.shape[0])
    B = cfg["batch_size"]
    is_dp = sigma > 0

    if is_dp:
        ps_fn = make_private_loss(model, per_sample_loss, configure_attention_backend=False)
        optimizer = DPOptimizer(
            Adam(learning_rate=cfg["lr"]),
            l2_norm_clip=clip,
            noise_multiplier=sigma,
            target_delta=cfg["target_delta"],
            num_samples=N,
            compile=False,
        )
    else:
        def batch_loss(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1), reduction="mean",
            )
        loss_and_grad = nn.value_and_grad(model, batch_loss)
        optimizer = Adam(learning_rate=cfg["lr"])

    steps_per_epoch = max(1, N // B)
    log = {"losses": [], "epoch_losses": []}

    t0 = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        perm = np.random.permutation(N)
        epoch_losses = []
        for step in range(steps_per_epoch):
            idx = perm[step * B : (step + 1) * B]
            if len(idx) < B:
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

        mean_epoch = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        log["epoch_losses"].append(mean_epoch)
        log["losses"].extend(epoch_losses)
        print(f"  epoch {epoch+1}/{cfg['epochs']}  mean_loss={mean_epoch:.4f}")

    elapsed = time.perf_counter() - t0
    epsilon = float(optimizer.epsilon) if is_dp else None

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.savez(os.path.join(output_dir, "adapter.npz"), **adapter_weights)

    meta = {
        **cfg,
        "lora_keys": list(cfg["lora_keys"]),
        "n_train": N,
        "elapsed_s": round(elapsed, 1),
        "final_loss": log["epoch_losses"][-1] if log["epoch_losses"] else None,
        "epsilon": epsilon,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(output_dir, "losses.json"), "w") as f:
        json.dump(log, f)

    print(f"  saved to {output_dir}  ε={epsilon}  elapsed={elapsed:.1f}s")
    return model, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=list(SETTINGS.keys()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, help="Path to split manifest JSON")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL (one text per line)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CFG, **SETTINGS[args.setting]}
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["seed"] = args.seed
    cfg["setting"] = args.setting

    corpus = []
    with open(args.corpus) as f:
        for line in f:
            obj = json.loads(line)
            corpus.append(obj["text"])

    split = load_split(args.split)
    member_ids = split["member_ids"]

    from mlx_lm import load as load_model
    _, tokenizer = load_model(cfg["model_name"])

    member_texts = [corpus[i] for i in member_ids]
    train_x, train_y = tokenize_corpus(tokenizer, member_texts, cfg["seq_len"])
    mx.eval(train_x, train_y)

    print(f"Training: setting={args.setting}, N={len(member_ids)}, epochs={cfg['epochs']}")
    train(cfg, train_x, train_y, args.output)


if __name__ == "__main__":
    main()
