"""Model compatibility checking for per-sample gradient computation."""

import mlx.nn as nn

_UNSUPPORTED_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

_QUANTIZED_MODULES = (
    nn.QuantizedLinear,
    nn.QuantizedEmbedding,
)


class UnsupportedModuleError(ValueError):
    pass


def check_model(model: nn.Module) -> None:
    """Check that a model's modules are compatible with vmap(grad).

    Rejects:
    - Trainable convolution modules (Pad vmap is NYI).
    - Any quantized modules in the forward path (QuantizedMatmul vmap is NYI).
      Quantized modules fail even when frozen because vmap traces the full
      forward pass.

    Raises:
        UnsupportedModuleError: If an incompatible module is found.
    """
    for name, module in model.named_modules():
        if isinstance(module, _UNSUPPORTED_MODULES) and module.trainable_parameters():
            raise UnsupportedModuleError(
                f"Module '{name}' ({type(module).__name__}) has trainable parameters "
                f"but is not supported for per-sample gradients. "
                f"The stock convolution backward-weight path fails under vmap "
                f"(Pad vmap is NYI in MLX {_mlx_version()}). "
                f"Options: (1) freeze this layer, (2) replace with a lowered "
                f"convolution implementation, or (3) upgrade MLX if a fix is available."
            )
        if isinstance(module, _QUANTIZED_MODULES):
            raise UnsupportedModuleError(
                f"Module '{name}' ({type(module).__name__}) uses quantized weights. "
                f"QuantizedMatmul does not support vmap (MLX {_mlx_version()}). "
                f"Use a non-quantized (bf16/fp16/fp32) model for DP training."
            )


def _mlx_version() -> str:
    try:
        import mlx.core as mx
        return mx.__version__
    except Exception:
        return "unknown"
