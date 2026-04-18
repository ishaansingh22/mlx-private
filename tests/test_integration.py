"""Integration tests for DPOptimizer and end-to-end DP-SGD.

These tests require MNIST data in /tmp/mnist. Run with:
    pytest -m mnist
"""

import math
import os
import gzip
import struct
import time

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx.optimizers import SGD, Adam

from private_mlx import make_private_loss, DPOptimizer

pytestmark = pytest.mark.mnist


MNIST_PATH = "/tmp/mnist"


def _load_mnist():
    def read_images(f):
        with gzip.open(os.path.join(MNIST_PATH, f)) as gz:
            _, n, r, c = struct.unpack(">4I", gz.read(16))
            return np.frombuffer(gz.read(), dtype=np.uint8).reshape(n, r * c).astype(np.float32) / 255.0

    def read_labels(f):
        with gzip.open(os.path.join(MNIST_PATH, f)) as gz:
            struct.unpack(">2I", gz.read(8))
            return np.frombuffer(gz.read(), dtype=np.uint8)

    return (
        mx.array(read_images("train-images-idx3-ubyte.gz")),
        mx.array(read_labels("train-labels-idx1-ubyte.gz")),
        mx.array(read_images("t10k-images-idx3-ubyte.gz")),
        mx.array(read_labels("t10k-labels-idx1-ubyte.gz")),
    )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def _loss_fn(model, x, y):
    logits = model(x[None, ...])
    return nn.losses.cross_entropy(logits, y[None], reduction="mean")


def _accuracy(model, x_test, y_test):
    logits = model(x_test)
    preds = mx.argmax(logits, axis=1)
    return mx.mean(preds == y_test).item() * 100


class TestCompileRandomness:

    def test_compiled_steps_produce_different_noise(self):
        """If mx.random.state isn't tracked, noise freezes. This catches that."""
        mx.random.seed(0)
        model = MLP()
        mx.eval(model.parameters())

        B = 8
        x = mx.random.normal((B, 784))
        y = mx.random.randint(0, 10, (B,))
        mx.eval(x, y)

        ps_fn = make_private_loss(model, _loss_fn)

        dp_opt = DPOptimizer(
            SGD(learning_rate=0.01), l2_norm_clip=1.0,
            noise_multiplier=10.0,  # high noise to make differences detectable
            target_delta=1e-5, num_samples=100, compile=True,
        )

        weight_snapshots = []
        for _ in range(10):
            grads = ps_fn(x, y)
            mx.eval(grads)
            dp_opt.step(model, grads)
            w = mx.array(model.fc1.weight)
            mx.eval(w)
            weight_snapshots.append(w)

        deltas = []
        for i in range(1, len(weight_snapshots)):
            d = weight_snapshots[i] - weight_snapshots[i - 1]
            deltas.append(d)

        unique_count = 0
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                if not mx.array_equal(deltas[i], deltas[j]).item():
                    unique_count += 1

        total_pairs = len(deltas) * (len(deltas) - 1) // 2
        assert unique_count == total_pairs, (
            f"Only {unique_count}/{total_pairs} delta pairs are unique. "
            f"Noise may be frozen across compiled steps."
        )

    def test_accountant_tracks_outside_compile(self):
        mx.random.seed(1)
        model = MLP()
        mx.eval(model.parameters())
        x = mx.random.normal((4, 784))
        y = mx.random.randint(0, 10, (4,))
        mx.eval(x, y)

        ps_fn = make_private_loss(model, _loss_fn)
        dp_opt = DPOptimizer(
            SGD(learning_rate=0.01), l2_norm_clip=1.0,
            noise_multiplier=1.0, target_delta=1e-5,
            num_samples=100, compile=True,
        )

        prev_eps = 0.0
        for step in range(20):
            grads = ps_fn(x, y)
            mx.eval(grads)
            dp_opt.step(model, grads)

        assert dp_opt.num_steps == 20
        assert dp_opt.epsilon > 0

    def test_epsilon_increases_monotonically_under_compile(self):
        """ε should increase with each step, not freeze."""
        mx.random.seed(2)
        model = MLP()
        mx.eval(model.parameters())
        x = mx.random.normal((8, 784))
        y = mx.random.randint(0, 10, (8,))
        mx.eval(x, y)

        ps_fn = make_private_loss(model, _loss_fn)
        dp_opt = DPOptimizer(
            SGD(learning_rate=0.01), l2_norm_clip=1.0,
            noise_multiplier=1.0, target_delta=1e-5,
            num_samples=100, compile=True,
        )

        epsilons = []
        for _ in range(10):
            grads = ps_fn(x, y)
            mx.eval(grads)
            dp_opt.step(model, grads)
            epsilons.append(dp_opt.epsilon)

        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1] - 1e-10, (
                f"ε not monotonic: step {i} ε={epsilons[i]}, prev={epsilons[i-1]}"
            )


class TestAggressiveClipping:

    def test_extreme_clip_zero_noise(self):
        """C=0.01, σ=0: every sample gets aggressively clipped, no noise.

        Validates clipping math independent of noise. Model should still
        train (slowly) since gradients retain direction.
        """
        mx.random.seed(10)
        model = MLP()
        mx.eval(model.parameters())

        x_train, y_train, x_test, y_test = _load_mnist()

        ps_fn = make_private_loss(model, _loss_fn)
        dp_opt = DPOptimizer(
            SGD(learning_rate=1.0),  # high lr to compensate for tiny grads
            l2_norm_clip=0.01, noise_multiplier=0.0,
            target_delta=1e-5, num_samples=60000, compile=False,
        )

        B = 256
        n = x_train.shape[0]
        steps = 3 * (n // B)  # 3 epochs

        losses = []
        for step in range(steps):
            idx = np.random.randint(0, n, (B,))
            xb = x_train[mx.array(idx)]
            yb = y_train[mx.array(idx)]

            grads = ps_fn(xb, yb)
            mx.eval(grads)
            dp_opt.step(model, grads)
            mx.eval(model.parameters())

            if step % 100 == 0:
                logits = model(xb)
                loss = nn.losses.cross_entropy(logits, yb, reduction="mean")
                mx.eval(loss)
                losses.append(loss.item())

        # Loss should decrease even with aggressive clipping
        assert losses[-1] < losses[0], (
            f"Loss didn't decrease with aggressive clipping: "
            f"start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

        acc = _accuracy(model, x_test, y_test)
        # Even with C=0.01, should get above random (>20%) after 3 epochs
        assert acc > 20, f"Accuracy too low with aggressive clipping: {acc:.1f}%"


class TestMNISTReproduction:

    def _train_dp(self, seed, epochs=5, B=256, lr=0.25, C=1.0, sigma=1.0, compile=False):
        mx.random.seed(seed)
        np.random.seed(seed)

        x_train, y_train, x_test, y_test = _load_mnist()
        model = MLP()
        mx.eval(model.parameters())

        ps_fn = make_private_loss(model, _loss_fn)
        dp_opt = DPOptimizer(
            SGD(learning_rate=lr), l2_norm_clip=C,
            noise_multiplier=sigma, target_delta=1e-5,
            num_samples=60000, compile=compile,
        )

        n = x_train.shape[0]
        steps_per_epoch = n // B

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            for step in range(steps_per_epoch):
                idx = perm[step * B : (step + 1) * B]
                xb = x_train[mx.array(idx)]
                yb = y_train[mx.array(idx)]

                grads = ps_fn(xb, yb)
                mx.eval(grads)
                dp_opt.step(model, grads)
                mx.eval(model.parameters())

        acc = _accuracy(model, x_test, y_test)
        eps = dp_opt.epsilon
        return acc, eps

    def test_matches_phase0(self):
        """Library should reproduce Phase 0 results: ~90% acc, ε≈1.13."""
        acc, eps = self._train_dp(seed=42, compile=False)
        assert acc > 88.0, f"Accuracy too low: {acc:.1f}% (expected ~90%)"
        assert abs(eps - 1.13) / 1.13 < 0.05, f"ε mismatch: {eps:.2f} (expected ~1.13)"

    def test_matches_phase0_compiled(self):
        """Same test with compile=True."""
        acc, eps = self._train_dp(seed=42, compile=True)
        assert acc > 88.0, f"Compiled accuracy too low: {acc:.1f}%"
        assert abs(eps - 1.13) / 1.13 < 0.05, f"Compiled ε mismatch: {eps:.2f}"

    def test_zero_noise_recovers_baseline(self):
        """σ=0, large C: should match non-DP training within ~2pp."""
        mx.random.seed(42)
        np.random.seed(42)

        x_train, y_train, x_test, y_test = _load_mnist()
        B, epochs = 256, 5
        n = x_train.shape[0]
        steps_per_epoch = n // B

        model_base = MLP()
        mx.eval(model_base.parameters())
        opt_base = SGD(learning_rate=0.1)

        def batch_loss(model, x, y):
            return nn.losses.cross_entropy(model(x), y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model_base, batch_loss)

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            for step in range(steps_per_epoch):
                idx = perm[step * B : (step + 1) * B]
                xb = x_train[mx.array(idx)]
                yb = y_train[mx.array(idx)]
                _, grads = loss_and_grad(model_base, xb, yb)
                opt_base.update(model_base, grads)
                mx.eval(model_base.parameters(), opt_base.state)

        baseline_acc = _accuracy(model_base, x_test, y_test)

        dp_acc, _ = self._train_dp(seed=42, C=1e10, sigma=0.0, lr=0.1)

        gap = abs(baseline_acc - dp_acc)
        assert gap < 2.0, (
            f"Zero-noise DP should match baseline: "
            f"baseline={baseline_acc:.1f}%, dp={dp_acc:.1f}%, gap={gap:.1f}pp"
        )
