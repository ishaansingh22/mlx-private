"""LoRA-scale DP-SGD validation: Qwen2.5-0.5B with DP training.

Verifies: no OOM, loss decreases, base weights frozen, ε reasonable.
Requires mlx-lm and downloads model weights. Run with:
    pytest -m lora
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx.optimizers import Adam

from private_mlx import make_private_loss, DPOptimizer, patch_model_for_dp

pytestmark = pytest.mark.lora


def _load_lora_model():
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
    model.freeze()
    lora_config = {
        "rank": 8, "scale": 20.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    linear_to_lora_layers(model, num_layers=4, config=lora_config)
    mx.eval(model.parameters())
    patch_model_for_dp(model)
    return model, tokenizer


def _lm_loss(model, x, y):
    logits = model(x[None, :])
    return nn.losses.cross_entropy(logits[0], y, reduction="mean")


def test_lora_dp_sgd():
    model, tokenizer = _load_lora_model()

    trainable = tree_flatten(model.trainable_parameters())
    n_trainable = sum(p.size for _, p in trainable)
    n_total = sum(p.size for _, p in tree_flatten(model.parameters()))

    base_snapshot = {}
    for name, p in tree_flatten(model.parameters()):
        if "lora" not in name:
            base_snapshot[name] = mx.array(p)
    mx.eval(list(base_snapshot.values()))

    seq_len = 32
    B = 4
    N_total = 200
    vocab_size = model.model.embed_tokens.weight.shape[0]

    mx.random.seed(42)
    np.random.seed(42)
    data_x = mx.random.randint(0, vocab_size, (N_total, seq_len))
    data_y = mx.random.randint(0, vocab_size, (N_total, seq_len))
    mx.eval(data_x, data_y)

    ps_fn = make_private_loss(model, _lm_loss)
    dp_opt = DPOptimizer(
        Adam(learning_rate=1e-4),
        l2_norm_clip=1.0, noise_multiplier=1.0,
        target_delta=1e-5, num_samples=N_total,
        compile=False,
    )

    losses = []
    for step in range(50):
        idx = np.random.randint(0, N_total, (B,))
        xb = data_x[mx.array(idx)]
        yb = data_y[mx.array(idx)]

        grads = ps_fn(xb, yb)
        mx.eval(grads)

        if step % 10 == 0:
            logits = model(xb)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.reshape(-1), reduction="mean",
            )
            mx.eval(loss)
            losses.append(loss.item())

        dp_opt.step(model, grads)
        mx.eval(model.parameters())

    logits = model(data_x[:B])
    final_loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        data_y[:B].reshape(-1), reduction="mean",
    )
    mx.eval(final_loss)
    losses.append(final_loss.item())

    eps = dp_opt.epsilon

    print(f"\n--- LoRA DP-SGD Results ---")
    print(f"Trainable: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.3f}%)")
    print(f"Loss trajectory: {[f'{l:.3f}' for l in losses]}")
    print(f"ε after 50 steps: {eps:.4f}")
    print(f"Steps: {dp_opt.num_steps}")

    assert losses[-1] < losses[0], (
        f"Loss didn't decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )

    changed_base = []
    for name, p in tree_flatten(model.parameters()):
        if name in base_snapshot:
            if not mx.array_equal(p, base_snapshot[name]).item():
                changed_base.append(name)

    assert len(changed_base) == 0, (
        f"Base (non-LoRA) weights changed: {changed_base[:5]}"
    )

    assert eps > 0, "ε should be positive after training"
    assert eps < 100, f"ε suspiciously large: {eps:.2f}"


if __name__ == "__main__":
    test_lora_dp_sgd()
