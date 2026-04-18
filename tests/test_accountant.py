"""Tests for private_mlx.accountant.

Cross-validates against Google's dp-accounting library (test-only dependency).
"""

import math
import itertools
import pytest

from private_mlx.accountant import (
    RDPAccountant,
    compute_rdp_poisson_subsampled_gaussian,
    rdp_to_epsilon,
    DEFAULT_ORDERS,
)

dp_accounting = pytest.importorskip("dp_accounting")
from dp_accounting import (
    NeighboringRelation,
    PoissonSampledDpEvent,
    GaussianDpEvent,
    SelfComposedDpEvent,
)
from dp_accounting.rdp import RdpAccountant as RefRdpAccountant


def _ref_epsilon(sigma: float, q: float, steps: int, delta: float) -> float:
    """Compute ε using Google's dp-accounting as ground truth."""
    accountant = RefRdpAccountant(
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE
    )
    event = SelfComposedDpEvent(
        event=PoissonSampledDpEvent(
            sampling_probability=q,
            event=GaussianDpEvent(noise_multiplier=sigma),
        ),
        count=steps,
    )
    accountant.compose(event)
    return accountant.get_epsilon(target_delta=delta)


_SIGMAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
_QS = [0.0001, 0.001, 0.01, 0.1, 0.5]
_STEPS = [10, 100, 1000, 10000]
_DELTAS = [1e-5, 1e-7, 1e-9]

_GRID = list(itertools.product(_SIGMAS, _QS, _STEPS, _DELTAS))


class TestCrossValidation:

    @pytest.mark.parametrize("sigma,q,steps,delta", _GRID)
    def test_epsilon_matches_reference(self, sigma, q, steps, delta):
        """Our ε must match dp-accounting within 0.5% relative error."""
        ref_eps = _ref_epsilon(sigma, q, steps, delta)

        acc = RDPAccountant(target_delta=delta)
        acc.step(noise_multiplier=sigma, sample_rate=q, num_steps=steps)
        our_eps = acc.epsilon

        # Both inf
        if math.isinf(ref_eps) and math.isinf(our_eps):
            return
        # Both zero
        if ref_eps == 0 and our_eps == 0:
            return
        # One inf, other not
        if math.isinf(ref_eps) != math.isinf(our_eps):
            pytest.fail(
                f"Inf mismatch: ours={our_eps}, ref={ref_eps} "
                f"(σ={sigma}, q={q}, T={steps}, δ={delta})"
            )

        # Relative error
        rel_err = abs(our_eps - ref_eps) / max(abs(ref_eps), 1e-12)
        assert rel_err < 0.005, (
            f"ε mismatch: ours={our_eps:.6f}, ref={ref_eps:.6f}, "
            f"rel_err={rel_err:.4f} "
            f"(σ={sigma}, q={q}, T={steps}, δ={delta})"
        )


class TestDrift:

    def test_epsilon_curve_matches_at_every_checkpoint(self):
        """ε should match reference at every checkpoint, not just final.

        Uses incremental composition (step-by-step between checkpoints) to
        verify that accumulation doesn't introduce drift.
        """
        sigma = 1.0
        q = 0.001
        delta = 1e-5
        checkpoints = [1, 10, 100, 500, 1000, 2000, 5000, 10000]

        our_acc = RDPAccountant(target_delta=delta)
        prev_ckpt = 0

        for ckpt in checkpoints:
            gap = ckpt - prev_ckpt
            our_acc.step(noise_multiplier=sigma, sample_rate=q, num_steps=gap)
            prev_ckpt = ckpt

            ref = _ref_epsilon(sigma, q, ckpt, delta)
            ours = our_acc.epsilon

            if ref == 0 and ours == 0:
                continue
            rel_err = abs(ours - ref) / max(abs(ref), 1e-12)
            assert rel_err < 0.005, (
                f"Drift at step {ckpt}: ours={ours:.6f}, ref={ref:.6f}, "
                f"rel_err={rel_err:.4f}"
            )


class TestMonotonicity:

    def test_higher_sigma_lower_epsilon(self):
        """More noise (higher σ) → lower ε for same (q, T, δ)."""
        q, steps, delta = 0.01, 1000, 1e-5
        prev_eps = math.inf
        for sigma in [0.5, 1.0, 2.0, 5.0, 10.0]:
            acc = RDPAccountant(target_delta=delta)
            acc.step(noise_multiplier=sigma, sample_rate=q, num_steps=steps)
            eps = acc.epsilon
            assert eps <= prev_eps + 1e-10, (
                f"Non-monotonic: σ={sigma} gave ε={eps}, prev ε={prev_eps}"
            )
            prev_eps = eps

    def test_more_steps_higher_epsilon(self):
        """More composition steps → higher ε for same (σ, q, δ)."""
        sigma, q, delta = 1.0, 0.01, 1e-5
        prev_eps = 0.0
        for steps in [10, 100, 1000, 5000]:
            acc = RDPAccountant(target_delta=delta)
            acc.step(noise_multiplier=sigma, sample_rate=q, num_steps=steps)
            eps = acc.epsilon
            assert eps >= prev_eps - 1e-10, (
                f"Non-monotonic: T={steps} gave ε={eps}, prev ε={prev_eps}"
            )
            prev_eps = eps

    def test_higher_q_higher_epsilon(self):
        """Higher sampling rate → higher ε for same (σ, T, δ)."""
        sigma, steps, delta = 1.0, 1000, 1e-5
        prev_eps = 0.0
        for q in [0.001, 0.01, 0.1, 0.5]:
            acc = RDPAccountant(target_delta=delta)
            acc.step(noise_multiplier=sigma, sample_rate=q, num_steps=steps)
            eps = acc.epsilon
            assert eps >= prev_eps - 1e-10, (
                f"Non-monotonic: q={q} gave ε={eps}, prev ε={prev_eps}"
            )
            prev_eps = eps


class TestEdgeCases:

    def test_zero_steps(self):
        acc = RDPAccountant(target_delta=1e-5)
        assert acc.epsilon == 0.0
        assert acc.num_steps == 0

    def test_zero_sampling_rate(self):
        acc = RDPAccountant(target_delta=1e-5)
        acc.step(noise_multiplier=1.0, sample_rate=0.0)
        assert acc.epsilon == 0.0

    def test_full_batch(self):
        """q=1.0 should give the non-subsampled Gaussian mechanism RDP."""
        acc = RDPAccountant(target_delta=1e-5)
        for _ in range(100):
            acc.step(noise_multiplier=1.0, sample_rate=1.0)
        assert acc.epsilon > 0
        assert not math.isinf(acc.epsilon)

    def test_step_counter(self):
        acc = RDPAccountant(target_delta=1e-5)
        acc.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=42)
        assert acc.num_steps == 42

    def test_step_counter_incremental(self):
        acc = RDPAccountant(target_delta=1e-5)
        acc.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=10)
        acc.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=32)
        assert acc.num_steps == 42
