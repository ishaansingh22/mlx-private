"""Attention backend compatibility for per-sample gradients.

MLX 0.31.1 has a known crash path: ``mx.fast.scaled_dot_product_attention``
segfaults under ``vmap(grad)`` for GQA/MQA head geometries (H_q != H_kv).
This module provides a selective fallback to decomposed attention that keeps
MHA on the fused path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from typing import Callable

import mlx.core as mx
import mlx.nn as nn


_VALID_BACKEND_MODES = {"auto", "fast", "manual"}

# module_name -> original scaled_dot_product_attention function
_PATCHED_MODULES: dict[str, Callable] = {}


def _manual_scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache=None,
    scale: float = 1.0,
    mask=None,
    **kwargs,
) -> mx.array:
    """Manual attention path used as GQA fallback."""
    if cache is not None and hasattr(cache, "bits"):
        raise NotImplementedError(
            "manual SDPA fallback does not support quantized KV caches. "
            "Use a non-quantized model for DP training."
        )

    # GQA: replicate K/V heads to match Q heads
    if keys.shape[1] != queries.shape[1]:
        n_rep = queries.shape[1] // keys.shape[1]
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    scores = (queries @ keys.swapaxes(-1, -2)) * scale

    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            T = queries.shape[-2]
            causal = mx.triu(mx.full((T, T), float("-inf")), k=1)
            scores = scores + causal
        elif isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask

    scores = mx.softmax(scores, axis=-1)
    return scores @ values


def _call_original(
    fn: Callable,
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask,
    **kwargs,
):
    return fn(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
        **kwargs,
    )


def _make_auto_scaled_dot_product_attention(
    original_fn: Callable,
    canary: dict[tuple[int, int, int], bool] | None,
) -> Callable:
    """Dispatch to fused for safe paths and manual for known-bad GQA paths."""

    def _auto_sdpa(
        queries,
        keys,
        values,
        cache=None,
        scale: float = 1.0,
        mask=None,
        **kwargs,
    ):
        if cache is not None and hasattr(cache, "bits"):
            return _call_original(
                original_fn,
                queries,
                keys,
                values,
                cache,
                scale,
                mask,
                **kwargs,
            )

        h_q = int(queries.shape[1])
        h_kv = int(keys.shape[1])
        head_dim = int(queries.shape[-1])

        if h_q != h_kv:
            if canary is not None and canary.get((h_q, h_kv, head_dim), False):
                return _call_original(
                    original_fn,
                    queries,
                    keys,
                    values,
                    cache,
                    scale,
                    mask,
                    **kwargs,
                )
            return _manual_scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=scale,
                mask=mask,
                **kwargs,
            )

        return _call_original(
            original_fn,
            queries,
            keys,
            values,
            cache,
            scale,
            mask,
            **kwargs,
        )

    return _auto_sdpa


def _resolve_mode(mode: str | None) -> str:
    if mode is None:
        mode = os.getenv("MLX_PRIVATE_ATTENTION_BACKEND", "auto")
    mode = mode.lower().strip()
    if mode not in _VALID_BACKEND_MODES:
        raise ValueError(
            f"Invalid attention backend '{mode}'. "
            f"Expected one of: {sorted(_VALID_BACKEND_MODES)}"
        )
    return mode


def _infer_head_dim(attention_module) -> int:
    for name in ("head_dim", "_head_dim", "d_head"):
        if hasattr(attention_module, name):
            try:
                return int(getattr(attention_module, name))
            except Exception:
                pass
    if hasattr(attention_module, "scale"):
        try:
            scale = float(getattr(attention_module, "scale"))
            if scale > 0:
                return int(round(scale ** -2))
        except Exception:
            pass
    return 64


def _model_has_gqa_modules(model: nn.Module) -> bool:
    for _, module in model.named_modules():
        n_h = getattr(module, "n_heads", None)
        n_kv = getattr(module, "n_kv_heads", None)
        if n_h is not None and n_kv is not None and int(n_h) != int(n_kv):
            return True
    return False


def _collect_patch_targets(model: nn.Module):
    targets: dict[str, object] = {}
    geometries: set[tuple[int, int, int]] = set()
    has_gqa = False

    for _, module in model.named_modules():
        if not (hasattr(module, "n_heads") and hasattr(module, "n_kv_heads")):
            continue

        owning_mod_name = type(module).__module__
        owning_mod = sys.modules.get(owning_mod_name)
        if owning_mod is None or not hasattr(owning_mod, "scaled_dot_product_attention"):
            continue

        targets[owning_mod_name] = owning_mod

        n_heads = int(getattr(module, "n_heads"))
        n_kv_heads = int(getattr(module, "n_kv_heads"))
        if n_heads != n_kv_heads:
            has_gqa = True
            geometries.add((n_heads, n_kv_heads, _infer_head_dim(module)))

    return targets, geometries, has_gqa


def _run_canary_enabled(run_canary: bool | None) -> bool:
    if run_canary is not None:
        return run_canary
    value = os.getenv("MLX_PRIVATE_SDPA_CANARY", "0").lower().strip()
    return value in {"1", "true", "yes", "on"}


def _sdpa_vmap_canary(n_heads: int, n_kv_heads: int, head_dim: int) -> bool:
    dim = n_heads * head_dim
    code = f"""
import mlx.core as mx
H_q={n_heads}
H_kv={n_kv_heads}
D={head_dim}
DIM={dim}

w_q = mx.random.normal((DIM, H_q * D))
w_k = mx.random.normal((DIM, H_kv * D))
w_v = mx.random.normal((DIM, H_kv * D))
mx.eval(w_q, w_k, w_v)

def loss_fn(wq, xi):
    x = xi[None, :]
    B, L, _ = x.shape
    q = (x @ wq).reshape(B, L, H_q, D).transpose(0, 2, 1, 3)
    k = (x @ w_k).reshape(B, L, H_kv, D).transpose(0, 2, 1, 3)
    v = (x @ w_v).reshape(B, L, H_kv, D).transpose(0, 2, 1, 3)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
    return mx.mean(out)

xb = mx.random.normal((2, 8, DIM))
mx.eval(xb)
vg = mx.vmap(mx.grad(loss_fn), in_axes=(None, 0))(w_q, xb)
mx.eval(vg)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    return proc.returncode == 0


def ensure_attention_backend_for_per_sample_grads(
    model: nn.Module,
    mode: str | None = None,
    *,
    run_canary: bool | None = None,
    strict: bool = False,
    warn: bool = True,
) -> nn.Module:
    """Configure attention backend for per-sample gradients.

    Modes:
    - ``auto`` (default): keep fused SDPA for MHA, fallback to manual SDPA for
      GQA/MQA. If canary is enabled and passes for a geometry, keep fused path.
    - ``fast``: force fused SDPA everywhere (may segfault on GQA under vmap-grad).
    - ``manual``: force manual SDPA everywhere for patchable modules.
    """
    backend = _resolve_mode(mode)
    targets, geometries, has_gqa = _collect_patch_targets(model)

    if not targets:
        has_any_gqa = _model_has_gqa_modules(model)
        if strict:
            raise ValueError(
                "No attention modules with scaled_dot_product_attention found. "
                "Ensure the model uses the mlx-lm base attention helper."
            )
        if has_any_gqa and warn:
            warnings.warn(
                "Model appears to have GQA/MQA attention (n_heads != n_kv_heads) "
                "but no patchable scaled_dot_product_attention was found. "
                "The fused SDPA path may hang or crash under vmap on MLX 0.31.x "
                "(see https://github.com/ml-explore/mlx/issues/3383). "
                "If you hit a hang/crash, use patch_model_for_dp() explicitly or "
                "replace mx.fast.scaled_dot_product_attention with decomposed attention.",
                stacklevel=2,
            )
        return model

    canary_results = None
    if backend == "auto" and has_gqa and _run_canary_enabled(run_canary):
        canary_results = {
            geom: _sdpa_vmap_canary(*geom)
            for geom in sorted(geometries)
        }

    for module_name, owning_mod in targets.items():
        current = getattr(owning_mod, "scaled_dot_product_attention")
        original = _PATCHED_MODULES.get(module_name, current)
        _PATCHED_MODULES[module_name] = original

        if backend == "fast":
            setattr(owning_mod, "scaled_dot_product_attention", original)
            continue

        if backend == "manual":
            setattr(owning_mod, "scaled_dot_product_attention", _manual_scaled_dot_product_attention)
            continue

        # auto mode
        if not has_gqa:
            setattr(owning_mod, "scaled_dot_product_attention", original)
        else:
            setattr(
                owning_mod,
                "scaled_dot_product_attention",
                _make_auto_scaled_dot_product_attention(original, canary_results),
            )

    if warn:
        if backend == "fast" and has_gqa:
            warnings.warn(
                "Using fused SDPA backend with GQA under per-sample gradients. "
                "This may segfault on MLX 0.31.1. "
                "Set MLX_PRIVATE_ATTENTION_BACKEND=auto or manual to avoid crashes.",
                stacklevel=2,
            )
        elif backend == "auto" and has_gqa:
            if canary_results is None:
                warnings.warn(
                    "Detected GQA attention modules. Using selective manual SDPA "
                    "fallback for per-sample gradients.",
                    stacklevel=2,
                )
            else:
                failed = [k for k, ok in canary_results.items() if not ok]
                if failed:
                    warnings.warn(
                        "SDPA canary failed for GQA geometries "
                        f"{failed}. Using manual fallback for those paths.",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "SDPA canary passed for detected GQA geometries. "
                        "Keeping fused SDPA in auto mode.",
                        stacklevel=2,
                    )
        elif backend == "manual":
            warnings.warn(
                "Using manual SDPA backend for all patchable attention modules.",
                stacklevel=2,
            )

    return model


def patch_model_for_dp(model: nn.Module) -> nn.Module:
    """Backward-compatible explicit manual backend patch."""
    return ensure_attention_backend_for_per_sample_grads(
        model,
        mode="manual",
        strict=True,
        warn=False,
    )


def unpatch_model_for_dp(model: nn.Module) -> nn.Module:
    """Restore original fused attention backend for patched modules."""
    targets, _, _ = _collect_patch_targets(model)
    for module_name, owning_mod in targets.items():
        if module_name in _PATCHED_MODULES:
            setattr(owning_mod, "scaled_dot_product_attention", _PATCHED_MODULES[module_name])
    return model
