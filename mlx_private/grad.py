"""Per-sample gradient computation via vmap(grad)."""

from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from ._check import check_model
from ._patch import ensure_attention_backend_for_per_sample_grads


def make_private_loss(
    model: nn.Module,
    loss_fn: Callable,
    *,
    validate: bool = True,
    configure_attention_backend: bool = True,
    attention_backend_mode: str | None = None,
    run_attention_canary: bool | None = None,
) -> Callable:
    """Wrap a loss function to produce per-sample gradients.

    Frozen parameters are excluded from the gradient tree entirely
    (no memory allocated for them).

    Args:
        model: The model to differentiate. Only ``model.trainable_parameters()``
            receive gradients.
        loss_fn: ``loss_fn(model, x, y) -> scalar``. Must be decomposable into
            independent per-sample terms (no cross-sample dependencies like
            contrastive loss or batch normalization).
        validate: If True, check model for unsupported trainable modules at
            construction time.
        configure_attention_backend: If True, configure SDPA backend for
            per-sample gradients. For GQA/MQA attention on MLX 0.31.1 this
            applies a selective manual fallback to avoid SIGSEGV under
            ``vmap(grad)``.
        attention_backend_mode: One of ``"auto"``, ``"fast"``, ``"manual"``,
            or ``None`` to read ``MLX_PRIVATE_ATTENTION_BACKEND`` (default:
            ``"auto"``).
        run_attention_canary: Optional subprocess canary toggle for SDPA
            GQA safety probe. ``None`` defers to ``MLX_PRIVATE_SDPA_CANARY``.

    Returns:
        ``per_sample_grads(x, y) -> grads`` where each leaf in *grads* has a
        leading batch dimension ``(B, *param_shape)``.
    """
    if validate:
        check_model(model)
    if configure_attention_backend:
        ensure_attention_backend_for_per_sample_grads(
            model,
            mode=attention_backend_mode,
            run_canary=run_attention_canary,
            strict=False,
            warn=True,
        )

    def _single_loss(params, *args, **kwargs):
        model.update(params)
        return loss_fn(model, *args, **kwargs)

    _single_grad = mx.grad(_single_loss)

    def per_sample_grads(*args, **kwargs):
        in_axes = (None,) + (0,) * len(args)
        return mx.vmap(_single_grad, in_axes=in_axes)(
            model.trainable_parameters(), *args, **kwargs
        )

    return per_sample_grads
