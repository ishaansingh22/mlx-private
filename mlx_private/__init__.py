"""mlx-private: Privacy-preserving training primitives for Apple MLX."""

from .grad import make_private_loss
from .clip import per_sample_global_norm, clip_and_aggregate, clip_and_aggregate_microbatched
from .accountant import RDPAccountant
from .optimizer import DPOptimizer
from ._check import check_model, UnsupportedModuleError
from ._patch import (
    ensure_attention_backend_for_per_sample_grads,
    patch_model_for_dp,
    unpatch_model_for_dp,
)

__all__ = [
    "make_private_loss",
    "per_sample_global_norm",
    "clip_and_aggregate",
    "clip_and_aggregate_microbatched",
    "RDPAccountant",
    "DPOptimizer",
    "check_model",
    "UnsupportedModuleError",
    "ensure_attention_backend_for_per_sample_grads",
    "patch_model_for_dp",
    "unpatch_model_for_dp",
]
