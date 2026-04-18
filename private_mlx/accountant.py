"""Rényi Differential Privacy accountant.

Ported from Google's dp-accounting library (Apache 2.0).
All numerical computation uses Python stdlib (math) — no scipy/numpy at runtime.
"""

import math
from typing import List, Tuple

# Standard RDP orders grid matching dp-accounting's default.
# Dense at low α (1.1–11.0 by 0.1), every integer 12–63,
# then sparse high orders for large-σ / few-steps regimes.
DEFAULT_ORDERS: Tuple[float, ...] = tuple(
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(11, 64))
    + [128, 256, 512, 1024]
)

_MAX_STEPS_LOG_A_FRAC = 1000


def _log_add(logx: float, logy: float) -> float:
    a, b = min(logx, logy), max(logx, logy)
    if a == -math.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_comb(n: float, k: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_erfc(x: float) -> float:
    r = math.erfc(x)
    if r == 0.0:
        # Laurent series for the tail of erfc when it underflows to 0.
        return (
            -math.log(math.pi) / 2
            - math.log(x)
            - x ** 2
            - 0.5 * x ** -2
            + 0.625 * x ** -4
            - 37.0 / 24.0 * x ** -6
            + 353.0 / 64.0 * x ** -8
        )
    return math.log(r)


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log(A_alpha) for integer alpha, 0 < q < 1."""
    log_a = -math.inf
    log1mq = math.log1p(-q)

    for i in range(alpha + 1):
        log_coef_i = _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * log1mq
        s = log_coef_i + (i * i - i) / (2 * sigma ** 2)
        log_a = _log_add(log_a, s)

    return log_a


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """log(A_alpha) for fractional alpha, 0 < q < 1."""
    log_a0 = -math.inf
    log_a1 = -math.inf
    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5
    log1mq = math.log1p(-q)

    last_s0 = last_s1 = -math.inf

    for i in range(_MAX_STEPS_LOG_A_FRAC):
        log_coef = _log_comb(alpha, i)
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * log1mq
        log_t1 = log_coef + j * math.log(q) + i * log1mq

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * sigma ** 2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma ** 2) + log_e1

        log_a0 = _log_add(log_a0, log_s0)
        log_a1 = _log_add(log_a1, log_s1)

        total = _log_add(log_a0, log_a1)

        if (
            log_s0 < last_s0
            and log_s1 < last_s1
            and max(log_s0, log_s1) < total - 30
        ):
            return total

        last_s0 = log_s0
        last_s1 = log_s1

    return math.inf


def compute_rdp_poisson_subsampled_gaussian(
    q: float,
    noise_multiplier: float,
    orders: Tuple[float, ...] = DEFAULT_ORDERS,
) -> List[float]:
    """Compute RDP of the Poisson-subsampled Gaussian mechanism.

    Args:
        q: Sampling probability (batch_size / dataset_size).
        noise_multiplier: σ = noise_std / l2_sensitivity.
        orders: RDP orders α to evaluate.

    Returns:
        List of RDP values, one per order.
    """
    if not 0 <= q <= 1:
        raise ValueError(f"Sampling rate must be in [0, 1]. Got {q}.")
    if noise_multiplier < 0:
        raise ValueError(f"Noise multiplier must be non-negative. Got {noise_multiplier}.")

    rdp = []
    for alpha in orders:
        if q == 0:
            rdp.append(0.0)
        elif math.isinf(alpha) or noise_multiplier == 0:
            rdp.append(math.inf)
        elif q == 1.0:
            rdp.append(alpha / (2 * noise_multiplier ** 2))
        else:
            if float(alpha).is_integer():
                log_a = _compute_log_a_int(q, noise_multiplier, int(alpha))
            else:
                log_a = _compute_log_a_frac(q, noise_multiplier, alpha)
            rdp.append(log_a / (alpha - 1))
    return rdp


def rdp_to_epsilon(
    orders: Tuple[float, ...],
    rdp_values: List[float],
    delta: float,
) -> Tuple[float, float]:
    """Convert RDP guarantees to (ε, δ)-DP.

    Uses the improved bound from Balle et al. (2020),
    https://arxiv.org/abs/2004.00010 Proposition 12.

    Returns:
        (epsilon, optimal_order)
    """
    if delta < 0:
        raise ValueError(f"Delta must be non-negative. Got {delta}.")
    if delta == 0:
        return (0.0, 0.0) if all(r == 0 for r in rdp_values) else (math.inf, 0.0)

    best_eps = math.inf
    best_order = 0.0

    for a, r in zip(orders, rdp_values):
        if r < 0:
            eps = 0.0
        elif delta ** 2 + math.expm1(-r) > 0:
            eps = 0.0
        elif a > 1.01:
            eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
        else:
            eps = math.inf

        if eps < best_eps:
            best_eps = eps
            best_order = a

    return max(0.0, best_eps), best_order


class RDPAccountant:
    """Tracks cumulative privacy loss via Rényi Differential Privacy.

    Usage::

        accountant = RDPAccountant(target_delta=1e-5)
        for step in training:
            accountant.step(noise_multiplier=1.0, sample_rate=batch_size/N)
        print(accountant.epsilon)
    """

    def __init__(self, target_delta: float, orders: Tuple[float, ...] = DEFAULT_ORDERS):
        self.target_delta = target_delta
        self._orders = orders
        self._cumulative_rdp = [0.0] * len(orders)
        self._steps = 0
        self._rdp_cache: dict[tuple[float, float], List[float]] = {}

    def step(self, noise_multiplier: float, sample_rate: float, num_steps: int = 1) -> None:
        """Record one or more identical DP-SGD steps.

        Args:
            noise_multiplier: σ for the Gaussian mechanism.
            sample_rate: Poisson sampling probability q = batch_size / N.
            num_steps: Number of identical steps to compose. Using
                ``num_steps=T`` is O(1) in T, vs calling ``step()`` T times
                which is O(T).
        """
        key = (noise_multiplier, sample_rate)
        rdp = self._rdp_cache.get(key)
        if rdp is None:
            rdp = compute_rdp_poisson_subsampled_gaussian(
                sample_rate, noise_multiplier, self._orders
            )
            self._rdp_cache[key] = rdp
        self._cumulative_rdp = [
            c + r * num_steps for c, r in zip(self._cumulative_rdp, rdp)
        ]
        self._steps += num_steps

    @property
    def epsilon(self) -> float:
        """Current ε at target δ."""
        eps, _ = rdp_to_epsilon(self._orders, self._cumulative_rdp, self.target_delta)
        return eps

    @property
    def num_steps(self) -> int:
        return self._steps
