"""Per-sample gradient clipping, noise addition, and aggregation."""

from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def per_sample_global_norm(per_sample_grads: Any) -> mx.array:
    """Compute the L2 norm of each sample's full gradient vector.

    Args:
        per_sample_grads: Pytree where each leaf has shape ``(B, *param_shape)``.

    Returns:
        1-D array of shape ``(B,)`` with each sample's global gradient norm.
    """
    flat = tree_flatten(per_sample_grads)
    if not flat:
        return mx.array([0.0])

    B = flat[0][1].shape[0]
    sq_norms = mx.zeros((B,))
    for _, g in flat:
        sq_norms = sq_norms + mx.sum(g.reshape(B, -1) ** 2, axis=1)
    return mx.sqrt(sq_norms)


def clip_and_aggregate(
    per_sample_grads: Any,
    l2_norm_clip: float,
    noise_multiplier: float,
) -> Any:
    """Clip per-sample gradients, sum, add calibrated noise, and average.

    Args:
        per_sample_grads: Pytree where each leaf has shape ``(B, *param_shape)``.
        l2_norm_clip: Maximum L2 norm ``C`` per sample.
        noise_multiplier: Noise scale ``σ`` relative to ``C``.

    Returns:
        Aggregated noisy gradient pytree (no batch dim) suitable for an
        optimizer update. Each leaf has the same shape as the model parameter.
    """
    flat = tree_flatten(per_sample_grads)
    if not flat:
        return per_sample_grads

    B = flat[0][1].shape[0]
    norms = per_sample_global_norm(per_sample_grads)

    # min(1, C / ||g||) per sample
    clip_factor = mx.minimum(1.0, l2_norm_clip / (norms + 1e-8))

    # Clip, sum across samples, add noise, average
    noised = []
    for key, g in flat:
        shape = (B,) + (1,) * (g.ndim - 1)
        clipped_sum = mx.sum(g * clip_factor.reshape(shape), axis=0)
        noise = mx.random.normal(clipped_sum.shape) * (l2_norm_clip * noise_multiplier)
        noised.append((key, (clipped_sum + noise) / B))

    return tree_unflatten(noised)


def clip_and_aggregate_microbatched(
    per_sample_grad_fn,
    batch_x: mx.array,
    batch_y: mx.array,
    l2_norm_clip: float,
    noise_multiplier: float,
    microbatch_size: int,
) -> Any:
    """Memory-efficient DP aggregation via microbatching.

    Processes the batch in chunks of ``microbatch_size``, clipping and
    accumulating without materializing all per-sample gradients at once.
    Noise is added once to the final aggregate. Produces the same result
    as ``clip_and_aggregate`` (given the same RNG state and samples).

    Args:
        per_sample_grad_fn: Function returned by ``make_private_loss``.
        batch_x: Full batch input, shape ``(B, ...)``.
        batch_y: Full batch labels, shape ``(B, ...)``.
        l2_norm_clip: Maximum L2 norm ``C`` per sample.
        noise_multiplier: Noise scale ``σ`` relative to ``C``.
        microbatch_size: Number of samples per microbatch ``m``.

    Returns:
        Aggregated noisy gradient pytree (no batch dim).
    """
    B = batch_x.shape[0]
    accum = None

    for start in range(0, B, microbatch_size):
        end = min(start + microbatch_size, B)
        mb_x = batch_x[start:end]
        mb_y = batch_y[start:end]

        mb_grads = per_sample_grad_fn(mb_x, mb_y)
        mb_flat = tree_flatten(mb_grads)

        mb_size = mb_x.shape[0]
        norms = per_sample_global_norm(mb_grads)
        clip_factor = mx.minimum(1.0, l2_norm_clip / (norms + 1e-8))

        if accum is None:
            accum = {}
            for key, g in mb_flat:
                shape = (mb_size,) + (1,) * (g.ndim - 1)
                accum[key] = mx.sum(g * clip_factor.reshape(shape), axis=0)
        else:
            for key, g in mb_flat:
                shape = (mb_size,) + (1,) * (g.ndim - 1)
                accum[key] = accum[key] + mx.sum(g * clip_factor.reshape(shape), axis=0)

        mx.eval(accum)

    # Noise once on the full aggregate, then average
    noised = []
    for key, clipped_sum in accum.items():
        noise = mx.random.normal(clipped_sum.shape) * (l2_norm_clip * noise_multiplier)
        noised.append((key, (clipped_sum + noise) / B))

    return tree_unflatten(noised)
