"""DP-SGD optimizer wrapping any MLX optimizer."""

from functools import partial
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from .clip import clip_and_aggregate, clip_and_aggregate_microbatched
from .accountant import RDPAccountant


def _batch_size(per_sample_grads: Any) -> int:
    flat = tree_flatten(per_sample_grads)
    if not flat:
        raise ValueError("Empty gradient pytree — nothing to clip.")
    return flat[0][1].shape[0]


class DPOptimizer:
    """Differentially private optimizer.

    Wraps any ``mlx.optimizers`` optimizer. Each ``step()`` call clips
    per-sample gradients to norm ``l2_norm_clip``, adds calibrated Gaussian
    noise, passes the result to the base optimizer, and tracks privacy spend
    via an RDP accountant.

    Args:
        base_optimizer: Any ``mlx.optimizers`` instance (SGD, Adam, etc.).
        l2_norm_clip: Max L2 norm ``C`` per sample.
        noise_multiplier: Noise scale ``σ`` relative to ``C``.
        target_delta: Target δ for (ε, δ)-DP accounting.
        num_samples: Total training set size ``N`` (for sampling rate ``q = B/N``).
        compile: If True, compile the array math (clip + noise + base update)
            for ~1.5–2x speedup. The accountant step (Python state) always
            runs outside the compiled region.
    """

    def __init__(
        self,
        base_optimizer,
        l2_norm_clip: float,
        noise_multiplier: float,
        target_delta: float,
        num_samples: int,
        compile: bool = True,
    ):
        self.base = base_optimizer
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_samples = num_samples
        self.accountant = RDPAccountant(target_delta)
        self._compile = compile
        self._compiled_step = None
        self._state = None

    def step(self, model, per_sample_grads) -> None:
        """Clip, noise, update, and account.

        Args:
            model: The ``nn.Module`` being trained.
            per_sample_grads: Pytree from ``make_private_loss``, each leaf
                shaped ``(B, *param_shape)``.
        """
        B = _batch_size(per_sample_grads)

        if self._compile:
            if self._compiled_step is None:
                self._state = [model.state, self.base.state, mx.random.state]
                clip = self.l2_norm_clip
                sigma = self.noise_multiplier
                base = self.base

                @partial(mx.compile, inputs=self._state, outputs=self._state)
                def _dp_update(psg):
                    noisy = clip_and_aggregate(psg, clip, sigma)
                    base.update(model, noisy)

                self._compiled_step = _dp_update

            self._compiled_step(per_sample_grads)
            mx.eval(self._state)
        else:
            noisy = clip_and_aggregate(per_sample_grads, self.l2_norm_clip, self.noise_multiplier)
            self.base.update(model, noisy)

        # Accountant runs outside compiled region (Python state mutation)
        self.accountant.step(self.noise_multiplier, B / self.num_samples)

    def step_microbatched(
        self,
        model,
        per_sample_grad_fn,
        batch_x: mx.array,
        batch_y: mx.array,
        microbatch_size: int,
    ) -> None:
        """Clip/noise/update with microbatched per-sample gradient evaluation."""
        if self._compile:
            raise ValueError("step_microbatched does not support compile=True.")

        B = int(batch_x.shape[0])

        noisy = clip_and_aggregate_microbatched(
            per_sample_grad_fn,
            batch_x,
            batch_y,
            self.l2_norm_clip,
            self.noise_multiplier,
            microbatch_size,
        )
        self.base.update(model, noisy)

        # Accountant runs outside compiled region (Python state mutation)
        self.accountant.step(self.noise_multiplier, B / self.num_samples)

    @property
    def epsilon(self) -> float:
        return self.accountant.epsilon

    @property
    def num_steps(self) -> int:
        return self.accountant.num_steps
