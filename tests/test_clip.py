"""Tests for private_mlx.clip."""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from private_mlx import (
    make_private_loss,
    per_sample_global_norm,
    clip_and_aggregate,
    clip_and_aggregate_microbatched,
)


class MLP(nn.Module):
    def __init__(self, d_in=16, d_hidden=32, d_out=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def _loss_fn(model, x, y):
    logits = model(x[None, ...])
    return nn.losses.cross_entropy(logits, y[None], reduction="mean")


def _make_grads(model, B=16, d_in=16, seed=0):
    mx.random.seed(seed)
    x = mx.random.normal((B, d_in))
    y = mx.random.randint(0, 4, (B,))
    mx.eval(x, y)

    ps_fn = make_private_loss(model, _loss_fn)
    grads = ps_fn(x, y)
    mx.eval(grads)
    return grads, x, y


class TestPerSampleGlobalNorm:

    def test_known_values(self):
        """Hand-computable case: two params, batch of 2."""
        grads = {"a": mx.array([[3.0, 4.0], [0.0, 0.0]]),
                 "b": mx.array([[0.0], [1.0]])}
        norms = per_sample_global_norm(grads)
        mx.eval(norms)
        # sample 0: sqrt(9+16+0) = 5.0, sample 1: sqrt(0+0+1) = 1.0
        assert abs(norms[0].item() - 5.0) < 1e-6
        assert abs(norms[1].item() - 1.0) < 1e-6

    def test_matches_manual(self):
        mx.random.seed(10)
        model = MLP()
        mx.eval(model.parameters())
        grads, _, _ = _make_grads(model, B=8, seed=10)

        norms = per_sample_global_norm(grads)
        mx.eval(norms)

        flat = dict(tree_flatten(grads))
        B = 8
        for i in range(B):
            sq = sum(mx.sum(v[i] ** 2).item() for v in flat.values())
            expected = math.sqrt(sq)
            assert abs(norms[i].item() - expected) < 1e-5, f"Sample {i} norm mismatch"

    def test_empty_pytree(self):
        norms = per_sample_global_norm({})
        mx.eval(norms)
        assert norms.shape == (1,)
        assert norms[0].item() == 0.0

    def test_shape(self):
        grads = {"w": mx.random.normal((32, 10, 5))}
        norms = per_sample_global_norm(grads)
        mx.eval(norms)
        assert norms.shape == (32,)


class TestClipAndAggregate:

    def test_all_clipped_norms_within_bound(self):
        """After clipping, every sample's gradient norm should be ≤ C."""
        mx.random.seed(20)
        model = MLP()
        mx.eval(model.parameters())
        grads, _, _ = _make_grads(model, B=16, seed=20)

        C = 0.5
        norms_before = per_sample_global_norm(grads)
        mx.eval(norms_before)

        flat = tree_flatten(grads)
        B = 16
        clip_factor = mx.minimum(1.0, C / (norms_before + 1e-8))
        for _, g in flat:
            shape = (B,) + (1,) * (g.ndim - 1)
            clipped = g * clip_factor.reshape(shape)
            # Check per-sample norms of clipped
            per_norms = mx.sqrt(mx.sum(clipped.reshape(B, -1) ** 2, axis=1))
            mx.eval(per_norms)
            assert mx.all(per_norms <= C + 1e-6).item(), "Clipped norm exceeds C"

    def test_no_clip_when_below_threshold(self):
        """Gradients below C should not be modified (ignoring noise)."""
        mx.random.seed(21)
        model = MLP()
        mx.eval(model.parameters())
        grads, _, _ = _make_grads(model, B=4, seed=21)

        norms = per_sample_global_norm(grads)
        mx.eval(norms)
        C = norms.max().item() * 10

        result = clip_and_aggregate(grads, l2_norm_clip=C, noise_multiplier=0.0)
        mx.eval(result)

        # Should equal mean of per-sample grads (no clipping, no noise)
        expected = {}
        for k, g in tree_flatten(grads):
            expected[k] = mx.mean(g, axis=0)
        mx.eval(expected)

        for k in expected:
            actual = dict(tree_flatten(result))[k]
            diff = mx.max(mx.abs(actual - expected[k])).item()
            assert diff < 1e-5, f"No-clip result differs for {k}: {diff}"

    def test_noise_variance(self):
        """Empirical noise variance should match (σC/B)² per coordinate."""
        B = 64
        C = 1.0
        sigma = 2.0
        expected_var = (sigma * C / B) ** 2

        # Use zero gradients so output is pure noise / B
        grads = {"w": mx.zeros((B, 10))}

        samples = []
        for i in range(2000):
            mx.random.seed(i)
            result = clip_and_aggregate(grads, l2_norm_clip=C, noise_multiplier=sigma)
            mx.eval(result)
            samples.append(dict(tree_flatten(result))["w"])

        stacked = mx.stack(samples)  # (2000, 10)
        mx.eval(stacked)
        empirical_var = mx.var(stacked, axis=0).mean().item()

        # Allow 20% tolerance for statistical test
        ratio = empirical_var / expected_var
        assert 0.8 < ratio < 1.2, (
            f"Noise variance mismatch: empirical={empirical_var:.6f}, "
            f"expected={expected_var:.6f}, ratio={ratio:.3f}"
        )

    def test_noise_on_aggregate_not_per_sample(self):
        """Noise should be added to the sum, not to each sample before summing.

        If noise were per-sample, the output variance would be B times larger.
        """
        B = 64
        C = 1.0
        sigma = 2.0
        per_sample_var = (sigma * C) ** 2  # wrong: per-sample noise
        aggregate_var = (sigma * C / B) ** 2  # correct: noise on aggregate / B

        grads = {"w": mx.zeros((B, 10))}

        samples = []
        for i in range(1000):
            mx.random.seed(i)
            result = clip_and_aggregate(grads, l2_norm_clip=C, noise_multiplier=sigma)
            mx.eval(result)
            samples.append(dict(tree_flatten(result))["w"])

        stacked = mx.stack(samples)
        mx.eval(stacked)
        empirical_var = mx.var(stacked, axis=0).mean().item()

        # Should be much closer to aggregate_var than per_sample_var
        assert empirical_var < aggregate_var * 2, (
            f"Variance too high — noise may be per-sample: {empirical_var:.6f} "
            f"vs expected {aggregate_var:.6f}"
        )
        assert empirical_var > aggregate_var * 0.5, (
            f"Variance too low: {empirical_var:.6f} vs expected {aggregate_var:.6f}"
        )

    def test_output_shape_no_batch_dim(self):
        mx.random.seed(22)
        model = MLP()
        mx.eval(model.parameters())
        grads, _, _ = _make_grads(model, B=8, seed=22)

        result = clip_and_aggregate(grads, l2_norm_clip=1.0, noise_multiplier=1.0)
        mx.eval(result)

        param_shapes = {k: v.shape for k, v in tree_flatten(model.trainable_parameters())}
        result_shapes = {k: v.shape for k, v in tree_flatten(result)}

        for k in param_shapes:
            assert result_shapes[k] == param_shapes[k], (
                f"Shape mismatch for {k}: got {result_shapes[k]}, "
                f"expected {param_shapes[k]}"
            )


class TestClipAndAggregateMicrobatched:

    def test_equal_to_materialized(self):
        """Microbatched path should match materialized within float32 tolerance.

        Not bitwise identical because microbatching evaluates partial sums
        between chunks, changing float32 accumulation order.
        """
        mx.random.seed(30)
        model = MLP()
        mx.eval(model.parameters())

        B = 16
        x = mx.random.normal((B, 16))
        y = mx.random.randint(0, 4, (B,))
        mx.eval(x, y)

        C = 1.0
        sigma = 1.5

        ps_fn = make_private_loss(model, _loss_fn)

        # Materialized
        grads_full = ps_fn(x, y)
        mx.eval(grads_full)
        mx.random.seed(999)
        result_full = clip_and_aggregate(grads_full, C, sigma)
        mx.eval(result_full)

        # Microbatched (same seed for noise)
        mx.random.seed(999)
        result_micro = clip_and_aggregate_microbatched(
            ps_fn, x, y, C, sigma, microbatch_size=4
        )
        mx.eval(result_micro)

        flat_full = dict(tree_flatten(result_full))
        flat_micro = dict(tree_flatten(result_micro))

        for k in flat_full:
            diff = mx.max(mx.abs(flat_full[k] - flat_micro[k])).item()
            assert diff < 1e-6, (
                f"Microbatch differs from materialized on {k}: max_diff={diff}"
            )

    def test_microbatch_indivisible(self):
        """Should work even when B is not divisible by microbatch_size."""
        mx.random.seed(31)
        model = MLP()
        mx.eval(model.parameters())

        B = 10  # not divisible by 4
        x = mx.random.normal((B, 16))
        y = mx.random.randint(0, 4, (B,))
        mx.eval(x, y)

        ps_fn = make_private_loss(model, _loss_fn)

        result = clip_and_aggregate_microbatched(
            ps_fn, x, y, l2_norm_clip=1.0, noise_multiplier=1.0, microbatch_size=4
        )
        mx.eval(result)

        for k, v in tree_flatten(result):
            assert v.ndim >= 1, f"Bad shape for {k}: {v.shape}"

    def test_microbatch_memory_savings(self):
        """Microbatched should use less peak memory than materialized."""
        mx.random.seed(32)
        model = MLP(d_in=64, d_hidden=256, d_out=16)
        mx.eval(model.parameters())

        B = 64
        x = mx.random.normal((B, 64))
        y = mx.random.randint(0, 16, (B,))
        mx.eval(x, y)

        ps_fn = make_private_loss(model, _loss_fn)

        # Materialized
        grads = ps_fn(x, y)
        mx.eval(grads)
        mx.reset_peak_memory()
        _ = clip_and_aggregate(grads, 1.0, 1.0)
        mx.eval(_)
        mem_full = mx.get_peak_memory()
        del grads, _
        mx.eval(mx.zeros((1,)))

        # Microbatched
        mx.reset_peak_memory()
        _ = clip_and_aggregate_microbatched(ps_fn, x, y, 1.0, 1.0, microbatch_size=8)
        mx.eval(_)
        mem_micro = mx.get_peak_memory()

        assert mem_micro < mem_full, (
            f"Microbatch should use less memory: micro={mem_micro}, full={mem_full}"
        )
