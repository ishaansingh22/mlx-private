"""Tests for private_mlx.grad and private_mlx._check."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from private_mlx import make_private_loss, check_model, UnsupportedModuleError


class MLP(nn.Module):
    def __init__(self, d_in=16, d_hidden=32, d_out=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)
        self._heads = heads
        self._head_dim = dim // heads
        self._scale = self._head_dim ** -0.5

    def __call__(self, x):
        B, T, D = x.shape
        q = self.wq(x).reshape(B, T, self._heads, self._head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self._heads, self._head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self._heads, self._head_dim).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.norm(self.wo(o))


class LoRAModel(nn.Module):
    def __init__(self, base_dim=128, rank=4, out_dim=4):
        super().__init__()
        self.base = nn.Linear(base_dim, base_dim)
        self.adapter_down = nn.Linear(base_dim, rank, bias=False)
        self.adapter_up = nn.Linear(rank, out_dim, bias=False)

    def __call__(self, x):
        h = nn.relu(self.base(x))
        return self.adapter_up(self.adapter_down(h))


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3)
        self.fc = nn.Linear(8, 4)

    def __call__(self, x):
        return self.fc(self.conv(x).mean(axis=(1, 2)))


def _loss_fn(model, x, y):
    logits = model(x[None, ...])
    return nn.losses.cross_entropy(logits, y[None], reduction="mean")


def _per_sample_norms(grads):
    flat = dict(tree_flatten(grads))
    B = next(iter(flat.values())).shape[0]
    norms = []
    for i in range(B):
        sq = sum(mx.sum(v[i] ** 2) for v in flat.values())
        norms.append(mx.sqrt(sq))
    result = mx.stack(norms)
    mx.eval(result)
    return result


def _loop_norms(model, loss_fn, x, y):
    def inner(params, xi, yi):
        model.update(params)
        return loss_fn(model, xi, yi)

    grad_fn = mx.grad(inner)
    norms = []
    for i in range(x.shape[0]):
        g = grad_fn(model.trainable_parameters(), x[i], y[i])
        mx.eval(g)
        sq = sum(mx.sum(v ** 2) for _, v in tree_flatten(g))
        norms.append(mx.sqrt(sq))
    result = mx.stack(norms)
    mx.eval(result)
    return result


class TestMakePrivateLoss:

    def test_mlp_shapes(self):
        mx.random.seed(0)
        model = MLP()
        mx.eval(model.parameters())

        ps_grad = make_private_loss(model, _loss_fn)
        x = mx.random.normal((8, 16))
        y = mx.random.randint(0, 4, (8,))
        mx.eval(x, y)

        grads = ps_grad(x, y)
        mx.eval(grads)

        flat = dict(tree_flatten(grads))
        assert set(flat.keys()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        for v in flat.values():
            assert v.shape[0] == 8

    def test_mlp_norm_correctness(self):
        mx.random.seed(1)
        model = MLP()
        mx.eval(model.parameters())

        B = 8
        x = mx.random.normal((B, 16))
        y = mx.random.randint(0, 4, (B,))
        mx.eval(x, y)

        ps_grad = make_private_loss(model, _loss_fn)
        grads = ps_grad(x, y)
        mx.eval(grads)

        vmap_n = _per_sample_norms(grads)
        loop_n = _loop_norms(model, _loss_fn, x, y)

        max_diff = mx.max(mx.abs(vmap_n - loop_n)).item()
        assert max_diff < 1e-5, f"Norm mismatch: {max_diff}"

    def test_transformer_block_norm_correctness(self):
        mx.random.seed(2)
        model = TransformerBlock(dim=64, heads=4)
        mx.eval(model.parameters())

        B, T, D = 4, 16, 64
        x = mx.random.normal((B, T, D))
        y = mx.random.normal((B, T, D))
        mx.eval(x, y)

        def loss_fn(model, x, y):
            return mx.mean((model(x[None, ...]) - y[None, ...]) ** 2)

        ps_grad = make_private_loss(model, loss_fn)
        grads = ps_grad(x, y)
        mx.eval(grads)

        vmap_n = _per_sample_norms(grads)
        loop_n = _loop_norms(model, loss_fn, x, y)

        max_diff = mx.max(mx.abs(vmap_n - loop_n)).item()
        assert max_diff < 1e-4, f"Transformer norm mismatch: {max_diff}"

    def test_frozen_params_excluded(self):
        mx.random.seed(3)
        model = LoRAModel()
        mx.eval(model.parameters())
        model.base.freeze()

        B = 4
        x = mx.random.normal((B, 128))
        y = mx.random.randint(0, 4, (B,))
        mx.eval(x, y)

        ps_grad = make_private_loss(model, _loss_fn)
        grads = ps_grad(x, y)
        mx.eval(grads)

        flat = dict(tree_flatten(grads))
        assert "base.weight" not in flat
        assert "base.bias" not in flat
        assert "adapter_down.weight" in flat
        assert "adapter_up.weight" in flat

    def test_frozen_params_memory_savings(self):
        """Trainable-only path should use far less memory than all-params."""
        mx.random.seed(4)
        model = LoRAModel(base_dim=512, rank=4, out_dim=4)
        mx.eval(model.parameters())
        model.base.freeze()

        B = 32
        x = mx.random.normal((B, 512))
        y = mx.random.randint(0, 4, (B,))
        mx.eval(x, y)

        # Trainable-only (what make_private_loss uses)
        ps_grad = make_private_loss(model, _loss_fn)
        mx.reset_peak_memory()
        g1 = ps_grad(x, y)
        mx.eval(g1)
        mem_trainable = mx.get_peak_memory()

        del g1
        mx.eval(mx.zeros((1,)))

        # All-params (what a naive implementation would use)
        def inner_all(params, xi, yi):
            model.update(params)
            return _loss_fn(model, xi, yi)

        vmap_all = mx.vmap(mx.grad(inner_all), in_axes=(None, 0, 0))
        mx.reset_peak_memory()
        g2 = vmap_all(model.parameters(), x, y)
        mx.eval(g2)
        mem_all = mx.get_peak_memory()

        assert mem_all > mem_trainable * 5, (
            f"Expected significant memory savings: trainable={mem_trainable}, all={mem_all}"
        )

    def test_model_state_preserved(self):
        mx.random.seed(5)
        model = MLP()
        mx.eval(model.parameters())

        before = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
        mx.eval(before)

        ps_grad = make_private_loss(model, _loss_fn)
        x = mx.random.normal((8, 16))
        y = mx.random.randint(0, 4, (8,))
        mx.eval(x, y)

        _ = ps_grad(x, y)
        mx.eval(_)

        after = dict(tree_flatten(model.trainable_parameters()))
        for k in before:
            assert mx.array_equal(before[k], after[k]).item(), f"State changed for {k}"

    def test_validate_false_skips_check(self):
        model = ConvModel()
        mx.eval(model.parameters())
        make_private_loss(model, _loss_fn, validate=False)


class TestCheckModel:

    def test_mlp_passes(self):
        model = MLP()
        check_model(model)

    def test_transformer_passes(self):
        model = TransformerBlock()
        mx.eval(model.parameters())
        check_model(model)

    def test_conv2d_trainable_fails(self):
        model = ConvModel()
        mx.eval(model.parameters())
        with pytest.raises(UnsupportedModuleError, match="Conv2d"):
            check_model(model)

    def test_conv2d_frozen_passes(self):
        model = ConvModel()
        mx.eval(model.parameters())
        model.conv.freeze()
        check_model(model)

    def test_conv1d_trainable_fails(self):
        class C(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(3, 8, kernel_size=3)
            def __call__(self, x):
                return self.conv(x)

        model = C()
        mx.eval(model.parameters())
        with pytest.raises(UnsupportedModuleError):
            check_model(model)

    def test_custom_module_passes(self):
        class Custom(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Linear(4, 4)
            def __call__(self, x):
                return self.b(nn.relu(self.a(x)))

        model = Custom()
        check_model(model)
