"""Personal writing demo: DP-LoRA fine-tune on your own text, then generate.

Usage:
    python demo.py --corpus my_writing.jsonl --output demo_run [--prompts prompts.txt]

Corpus format: one JSON object per line with a "text" field.
Prompts format: one prompt per line.
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import Adam
from mlx.utils import tree_flatten

from mlx_private import DPOptimizer, make_private_loss
from mlx_private._patch import ensure_attention_backend_for_per_sample_grads


DEFAULT_PROMPTS = [
    "Write a short paragraph about morning routines:",
    "Describe your favorite place to think:",
    "Explain why you started this project:",
]


def build_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    from mlx_lm import load as load_model
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load_model(model_name)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=4,
        config={
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        },
    )
    mx.eval(model.parameters())
    ensure_attention_backend_for_per_sample_grads(model, mode="auto", warn=False)
    return model, tokenizer


def tokenize_corpus(tokenizer, texts: list[str], seq_len: int = 64):
    all_x, all_y = [], []
    for text in texts:
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


def generate(model, tokenizer, prompt: str, max_tokens: int = 100, temp: float = 0.7):
    ids = tokenizer.encode(prompt)
    tokens = mx.array([ids])

    for _ in range(max_tokens):
        logits = model(tokens)
        next_logit = logits[:, -1, :]
        if temp > 0:
            next_logit = next_logit / temp
            probs = mx.softmax(next_logit, axis=-1)
            next_tok = mx.random.categorical(mx.log(probs))
        else:
            next_tok = mx.argmax(next_logit, axis=-1)
        mx.eval(next_tok)
        tok_id = int(next_tok.item())
        if tok_id == tokenizer.eos_token_id:
            break
        tokens = mx.concatenate([tokens, next_tok[:, None]], axis=1)

    return tokenizer.decode(tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="JSONL with 'text' field per line")
    parser.add_argument("--output", default="demo_run")
    parser.add_argument("--prompts", help="Text file with one prompt per line")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    texts = []
    with open(args.corpus) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    N = len(texts)
    print(f"Loaded {N} examples from {args.corpus}")

    model, tokenizer = build_model()
    train_x, train_y = tokenize_corpus(tokenizer, texts)
    mx.eval(train_x, train_y)

    ps_fn = make_private_loss(model, per_sample_loss, configure_attention_backend=False)
    dp_opt = DPOptimizer(
        Adam(learning_rate=1e-4),
        l2_norm_clip=args.clip,
        noise_multiplier=args.noise_multiplier,
        target_delta=1e-5,
        num_samples=N,
        compile=False,
    )

    B = min(4, N)
    steps_per_epoch = max(1, N // B)

    for epoch in range(args.epochs):
        perm = np.random.permutation(N)
        epoch_losses = []
        for step in range(steps_per_epoch):
            idx = perm[step * B : (step + 1) * B]
            if len(idx) < B:
                continue
            xb = train_x[mx.array(idx)]
            yb = train_y[mx.array(idx)]
            grads = ps_fn(xb, yb)
            mx.eval(grads)
            dp_opt.step(model, grads)
            mx.eval(model.parameters())

            logits = model(xb)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.reshape(-1), reduction="mean",
            )
            mx.eval(loss)
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"  epoch {epoch+1}/{args.epochs}  loss={mean_loss:.4f}")

    eps = dp_opt.epsilon
    print(f"  ε = {eps:.2f}")

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.savez(os.path.join(args.output, "adapter.npz"), **adapter_weights)

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]

    results = []
    print(f"\n--- Generations (ε={eps:.2f}) ---\n")
    for prompt in prompts:
        output = generate(model, tokenizer, prompt)
        results.append({"prompt": prompt, "output": output})
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")

    meta = {
        "n_train": N,
        "epochs": args.epochs,
        "noise_multiplier": args.noise_multiplier,
        "l2_norm_clip": args.clip,
        "epsilon": float(eps),
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "demo_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.output, "generations.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}/")


if __name__ == "__main__":
    main()
