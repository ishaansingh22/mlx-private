"""Fast regression: compiled DP noise must differ across steps.

Runs under plain ``pytest`` (no markers). Catches the critical bug where
``mx.compile`` traces ``mx.random.normal`` once and freezes the noise if
``mx.random.state`` is not in the captured state list.
"""

from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import SGD
from mlx.utils import tree_flatten

from private_mlx import make_private_loss, DPOptimizer, clip_and_aggregate


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def _loss_fn(model, x, y):
    logits = model(x[None, ...])
    return nn.losses.cross_entropy(logits, y[None], reduction="mean")


class TestCompiledNoiseFreshness:
    """Verify that mx.compile does not freeze Gaussian noise in the DP path."""

    def test_compiled_clip_and_aggregate_produces_fresh_noise(self):
        """Compiled clip_and_aggregate with tracked RNG state must produce
        different outputs on each call. Zero grads isolate pure noise."""
        mx.random.seed(42)

        B, D = 8, 10
        C, sigma = 1.0, 10.0
        zero_grads = {"w": mx.zeros((B, D))}
        mx.eval(zero_grads)

        # Mirror DPOptimizer's compile pattern: track mx.random.state
        state = [mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_aggregate(psg):
            return clip_and_aggregate(psg, C, sigma)

        samples = []
        for _ in range(100):
            result = compiled_aggregate(zero_grads)
            mx.eval(state)
            w = mx.array(result["w"])
            mx.eval(w)
            samples.append(w)

        # Consecutive outputs must differ
        for i in range(1, len(samples)):
            assert not mx.array_equal(samples[i], samples[i - 1]).item(), (
                f"Compiled noise frozen: step {i} identical to step {i-1}"
            )

        # Stronger check on a subset of all pairs
        n_unique = 0
        total_pairs = 0
        for i in range(20):
            for j in range(i + 1, 20):
                total_pairs += 1
                if not mx.array_equal(samples[i], samples[j]).item():
                    n_unique += 1
        assert n_unique == total_pairs, (
            f"Only {n_unique}/{total_pairs} sample pairs unique — "
            f"noise may be frozen across compiled steps."
        )

        # Noise scale: output = N(0, (σC)²) / B, so std per coord = σC/B
        expected_std = sigma * C / B
        stacked = mx.stack(samples)  # (100, D)
        mx.eval(stacked)
        empirical_std = mx.sqrt(mx.var(stacked, axis=0)).mean().item()
        ratio = empirical_std / expected_std
        assert 0.8 < ratio < 1.2, (
            f"Noise scale wrong: empirical std={empirical_std:.4f}, "
            f"expected={expected_std:.4f}, ratio={ratio:.3f}"
        )

    def test_compiled_noise_frozen_without_state_tracking(self):
        """Negative control: compiling WITHOUT tracking mx.random.state
        should freeze the noise, confirming the test mechanism works."""
        mx.random.seed(99)

        B, D = 8, 10
        C, sigma = 1.0, 10.0
        zero_grads = {"w": mx.zeros((B, D))}
        mx.eval(zero_grads)

        @mx.compile
        def compiled_no_state(psg):
            return clip_and_aggregate(psg, C, sigma)

        samples = []
        for _ in range(5):
            result = compiled_no_state(zero_grads)
            w = mx.array(result["w"])
            mx.eval(w)
            samples.append(w)

        # Without state tracking, all outputs should be identical
        for i in range(1, len(samples)):
            assert mx.array_equal(samples[i], samples[0]).item(), (
                "Negative control failed: noise differs even without state "
                "tracking — test mechanism may be invalid."
            )


class TestDPOptimizerCompileSmoke:
    """Compiled DPOptimizer smoke test with a tiny model, no data download."""

    def test_compiled_steps_fresh_noise_and_accounting(self):
        """20 compiled DP steps: noise varies, num_steps correct, epsilon monotonic."""
        mx.random.seed(0)
        model = TinyMLP()
        mx.eval(model.parameters())

        B = 8
        x = mx.random.normal((B, 16))
        y = mx.random.randint(0, 4, (B,))
        mx.eval(x, y)

        ps_fn = make_private_loss(model, _loss_fn)
        dp_opt = DPOptimizer(
            SGD(learning_rate=0.01),
            l2_norm_clip=1.0,
            noise_multiplier=10.0,
            target_delta=1e-5,
            num_samples=100,
            compile=True,
        )

        weight_snapshots = []
        epsilons = []
        for _ in range(20):
            grads = ps_fn(x, y)
            mx.eval(grads)
            dp_opt.step(model, grads)
            w = mx.array(model.fc1.weight)
            mx.eval(w)
            weight_snapshots.append(w)
            epsilons.append(dp_opt.epsilon)

        assert dp_opt.num_steps == 20

        # Weight deltas must differ across steps (fresh noise each time)
        deltas = [
            weight_snapshots[i] - weight_snapshots[i - 1]
            for i in range(1, len(weight_snapshots))
        ]
        for i in range(1, len(deltas)):
            assert not mx.array_equal(deltas[i], deltas[i - 1]).item(), (
                f"Weight delta frozen: step {i+1} identical to step {i}"
            )

        # Epsilon must increase monotonically
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1] - 1e-10, (
                f"Epsilon not monotonic: step {i} eps={epsilons[i]:.6f}, "
                f"prev={epsilons[i-1]:.6f}"
            )

        assert epsilons[-1] > 0

    def test_microbatched_step_matches_materialized_no_noise(self):
        """Microbatched DP step should match materialized step when σ=0."""
        mx.random.seed(7)
        model_full = TinyMLP()
        mx.eval(model_full.parameters())
        mx.random.seed(7)
        model_micro = TinyMLP()
        mx.eval(model_micro.parameters())

        B = 8
        x = mx.random.normal((B, 16))
        y = mx.random.randint(0, 4, (B,))
        mx.eval(x, y)

        ps_fn_full = make_private_loss(model_full, _loss_fn)
        ps_fn_micro = make_private_loss(model_micro, _loss_fn)
        opt_full = DPOptimizer(
            SGD(learning_rate=0.01),
            l2_norm_clip=1.0,
            noise_multiplier=0.0,
            target_delta=1e-5,
            num_samples=100,
            compile=False,
        )
        opt_micro = DPOptimizer(
            SGD(learning_rate=0.01),
            l2_norm_clip=1.0,
            noise_multiplier=0.0,
            target_delta=1e-5,
            num_samples=100,
            compile=False,
        )

        for _ in range(5):
            grads = ps_fn_full(x, y)
            mx.eval(grads)
            opt_full.step(model_full, grads)
            opt_micro.step_microbatched(
                model_micro,
                ps_fn_micro,
                x,
                y,
                microbatch_size=2,
            )
            mx.eval(model_full.parameters(), model_micro.parameters())

        full_params = dict(tree_flatten(model_full.parameters()))
        micro_params = dict(tree_flatten(model_micro.parameters()))
        for key in full_params:
            diff = mx.max(mx.abs(full_params[key] - micro_params[key])).item()
            assert diff < 1e-6, f"Mismatch on {key}: max_diff={diff}"

        assert opt_full.num_steps == opt_micro.num_steps == 5
