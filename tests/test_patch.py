"""Tests for selective attention backend fallback."""

import os
import importlib

import mlx.core as mx
import mlx.nn as nn

from private_mlx import ensure_attention_backend_for_per_sample_grads


class _FakeAttention:
    __module__ = "tests._fake_attention_mod"

    def __init__(self, n_heads, n_kv_heads):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.scale = 0.125


class _FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        for i, m in enumerate(self._modules):
            yield f"m{i}", m


def _install_fake_module():
    from private_mlx import _patch as patch_mod

    mod_name = "tests._fake_attention_mod"
    if mod_name in importlib.sys.modules:
        mod = importlib.sys.modules[mod_name]
    else:
        mod = type(importlib)("tests._fake_attention_mod")
        importlib.sys.modules[mod_name] = mod

    if not hasattr(mod, "_orig_sdpa"):
        def _orig_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kwargs):
            # Return sentinel shape-compatible tensor
            return values

        mod._orig_sdpa = _orig_sdpa

    # Clear previous patch bookkeeping for deterministic tests.
    patch_mod._PATCHED_MODULES.pop(mod_name, None)
    mod.scaled_dot_product_attention = mod._orig_sdpa
    return mod


def test_auto_mode_keeps_mha_fused():
    mod = _install_fake_module()
    model = _FakeModel([_FakeAttention(n_heads=4, n_kv_heads=4)])
    orig = mod.scaled_dot_product_attention

    ensure_attention_backend_for_per_sample_grads(model, mode="auto", warn=False)

    assert mod.scaled_dot_product_attention is orig


def test_auto_mode_wraps_gqa():
    mod = _install_fake_module()
    model = _FakeModel([_FakeAttention(n_heads=14, n_kv_heads=2)])
    orig = mod.scaled_dot_product_attention

    ensure_attention_backend_for_per_sample_grads(model, mode="auto", warn=False)

    assert mod.scaled_dot_product_attention is not orig


def test_manual_mode_always_patches():
    mod = _install_fake_module()
    model = _FakeModel([_FakeAttention(n_heads=4, n_kv_heads=4)])
    orig = mod.scaled_dot_product_attention

    ensure_attention_backend_for_per_sample_grads(model, mode="manual", warn=False)

    assert mod.scaled_dot_product_attention is not orig


def test_fast_mode_restores_original():
    mod = _install_fake_module()
    model = _FakeModel([_FakeAttention(n_heads=14, n_kv_heads=2)])
    orig = mod.scaled_dot_product_attention

    ensure_attention_backend_for_per_sample_grads(model, mode="manual", warn=False)
    assert mod.scaled_dot_product_attention is not orig

    ensure_attention_backend_for_per_sample_grads(model, mode="fast", warn=False)
    assert mod.scaled_dot_product_attention is orig


def test_env_override_mode():
    mod = _install_fake_module()
    model = _FakeModel([_FakeAttention(n_heads=14, n_kv_heads=2)])
    orig = mod.scaled_dot_product_attention

    old = os.getenv("MLX_PRIVATE_ATTENTION_BACKEND")
    os.environ["MLX_PRIVATE_ATTENTION_BACKEND"] = "manual"
    try:
        ensure_attention_backend_for_per_sample_grads(model, mode=None, warn=False)
        assert mod.scaled_dot_product_attention is not orig
    finally:
        if old is None:
            os.environ.pop("MLX_PRIVATE_ATTENTION_BACKEND", None)
        else:
            os.environ["MLX_PRIVATE_ATTENTION_BACKEND"] = old

