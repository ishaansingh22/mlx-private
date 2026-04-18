# private-mlx

Per-sample differentially private SGD (DP-SGD) for LoRA fine-tuning on Apple Silicon. All training runs on-device via MLX.

```python
from private_mlx import make_private_loss, DPOptimizer

dp_loss_fn = make_private_loss(model, loss_fn)
optimizer = DPOptimizer(base_optimizer, l2_norm_clip=1.0, noise_multiplier=1.1,
                        target_delta=1e-5, num_samples=60000)

for batch_x, batch_y in dataloader:
    grads = dp_loss_fn(batch_x, batch_y)
    mx.eval(grads)
    optimizer.step(model, grads)
    mx.eval(model.parameters())

print(f"ε = {optimizer.epsilon:.2f}")
```

## Why Differential Privacy?

You fine-tune a model on private data (emails, medical notes, user messages) and then publish it or let people query it. The concern: **can someone learn facts about the training data just from the model's behavior?**

**Membership inference** is the simplest version of this. An attacker is anyone who can run the model. They take a candidate text, feed it through, and check how well the model predicts it. Models assign each input a **loss**: lower means "the model fits this text surprisingly well," higher means "this text looks unfamiliar." If the model fits a text *too* well compared to similar texts it never trained on, that's a signal the text was in the training set. This is a **loss-threshold membership inference attack (MIA)**. Pick a cutoff, and everything below it gets flagged as "probably a training example."

This works because small models in standard training tend to **memorize** parts of their training data, so training examples look unnaturally easy to the model, even when overall accuracy looks fine.

**Differential privacy** is a training technique that limits how much any single example can influence the final model. Each gradient update is clipped per-sample and noised so no individual row leaves a strong fingerprint in the weights. The privacy budget **ε** (epsilon) quantifies the guarantee: lower ε means stronger privacy. The trade-off is some accuracy loss, which the results below measure.

**MIA AUC** (area under the ROC curve) measures how well the attacker separates members from non-members across all thresholds. 0.5 is random guessing, 1.0 is perfect detection. The goal of DP training is to push MIA AUC toward 0.5 while keeping task accuracy as high as possible.

## Membership Inference Results

Trained Qwen2.5-0.5B LoRA adapters across four benchmarks and ran loss-threshold membership inference attacks against each. The question throughout: can an attacker distinguish training examples from held-out examples by their loss?

### Canary Frontier

Synthetic corpus, 200 training examples with unique 8-character codes. Seed 42.

| Setting | ε | ROC-AUC | Bal. Accuracy | TPR @ 1% FPR |
|---|---|---|---|---|
| No DP | ∞ | **0.917** | **0.843** | **0.545** |
| DP (σ=0.5) | 15.43 | 0.507 | 0.525 | 0.005 |
| DP (σ=1.5) | 1.08 | 0.514 | 0.535 | 0.010 |

Without DP, the attacker identifies training examples at AUC 0.917. With DP, the attack falls to chance at both noise levels.

`python3 examples/canary_frontier.py`

### SST-2

Binary sentiment, forced-choice Yes/No, label-only loss mask. 2000 train / 800 test, balanced. 5 seeds.

| Setting | ε | Accuracy | MIA AUC |
|---|---|---|---|
| No DP | ∞ | **87.4 ± 0.9%** | 0.661 |
| DP (σ=0.5) | 8.76 | **82.5 ± 2.1%** | 0.507 |

Utility gap: 5.0pp ± 2.8pp (range 0.5–7.9pp across seeds). The per-seed variance is large — DP noise dominates the gap at this training budget. MIA AUC is near 0.50 on every seed.

`python3 examples/sst2_dp.py --seed 42 --epochs 2 --logical-batch-size 16 --microbatch-size 4 --n-train 2000 --n-test 800`

### IMDB

Longer reviews, label-only mask, microbatched DP updates. 1500 train / 1500 test. 3 seeds.

| Setting | ε | Accuracy | MIA AUC | Member Loss | NM Loss |
|---|---|---|---|---|---|
| No DP | ∞ | **78.0 ± 0.8%** | 0.926 | 0.025 | 0.811 |
| DP (σ=0.5) | 9.72 | **77.3 ± 1.7%** | 0.500 | 0.859 | 0.838 |

The non-DP model has 32x lower loss on members than nonmembers. This is memorization of the training set, detectable at AUC 0.926, even though accuracy looks normal. DP eliminates the gap entirely. On one seed, DP outperforms non-DP — the noise regularizes against overfitting at 5 epochs.

`python3 examples/imdb_dp.py --seed 42 --epochs 5`

### PubMedQA

500 real medical QA examples. The fine-tune does not learn the task at this scale — accuracy sits at the majority baseline. This benchmark tests whether DP prevents membership leakage when the model fails to generalize.

| Setting | ε | MIA AUC | Member Loss | NM Loss |
|---|---|---|---|---|
| No DP | ∞ | **0.861** | 0.045 | 0.497 |
| DP (σ=0.5) | 8.80 | 0.538 | 0.618 | 0.736 |
| DP (σ=1.5) | 0.47 | 0.538 | 0.566 | 0.680 |

It does. A model that learns nothing still leaks membership at AUC 0.861. DP closes the leak.

`python3 examples/pubmedqa_dp.py`

### Summary

| Benchmark | non_dp MIA AUC | dp_mid MIA AUC | Utility Gap |
|---|---|---|---|
| Canary | 0.917 | 0.507 | — |
| SST-2 (5 seeds) | 0.661 | 0.507 | 5.0 ± 2.8pp |
| IMDB (3 seeds) | 0.926 | 0.500 | 0.6 ± 2.4pp |
| PubMedQA | 0.861 | 0.538 | — |

These results are conservative. Post-fix per-sample gradient norms have median 2.34 and p90 29.8; the current clip `C=1.0` clips ~88% of samples. Raising the clip toward the median would recover utility without materially increasing privacy cost.

## Install

```bash
pip install -e .          # core library
pip install -e ".[test]"  # + pytest, dp-accounting
pip install -e ".[lora]"  # + mlx-lm for LoRA workflows
```

## Scope

**v0.1 supports:**

LoRA fine-tuning of decoder-only transformers on Apple Silicon via `mlx-lm`. Non-quantized (bf16/fp16/fp32) base models with frozen weights and trainable LoRA adapters. Qwen and Llama validated; other GQA/MQA architectures with the standard `mlx-lm` attention pattern should work. Single-device only.

**Not supported:**

Quantized base models (`QuantizedMatmul::vmap` is NYI in MLX). Full fine-tuning (memory = O(B × total_params); use LoRA). Ghost clipping or memory-efficient per-sample gradient approximations. Stock `nn.Conv2d` with padding under `vmap(grad)`. Cryptographically secure noise generation. Multi-device / distributed training.

## How It Works

`make_private_loss` wraps your loss with `mx.vmap(mx.grad(...))`. Only trainable (unfrozen) parameters receive gradients — no memory spent on frozen base weights.

For GQA models, the fused SDPA kernel is replaced with decomposed attention. MLX 0.31.1's `mx.fast.scaled_dot_product_attention` hangs under `vmap` when query and key/value head counts differ ([ml-explore/mlx#3383](https://github.com/ml-explore/mlx/issues/3383)). The fallback is selective: only GQA modules are patched; MHA stays on the fused path. ~1.45x attention overhead.

`DPOptimizer.step()` clips each sample's gradient to L2 norm C, sums, adds N(0, σ²C²) noise, averages, and delegates to the base optimizer. The RDP accountant tracks ε automatically. With `compile=True` (default), `mx.random.state` is captured as mutable compile state so Gaussian noise is resampled every step — verified by `test_compile_randomness.py`.

### Privacy Accounting

The RDP accountant assumes Poisson subsampling (each example included independently with probability q = B/N). The actual data loader uses fixed-size uniform sampling without replacement, which provides slightly weaker amplification. Reported ε is therefore conservative — actual privacy is at least as good as stated, but the bound is not tight. See Balle et al. 2018 for the distinction. This matches the accounting in Opacus.

## API

| Function | Purpose |
|---|---|
| `make_private_loss(model, loss_fn)` | Wrap loss with `vmap(grad)` for per-sample gradients |
| `DPOptimizer(base, C, σ, δ, N)` | DP optimizer with built-in RDP accountant |
| `RDPAccountant(δ)` | Standalone privacy accountant |
| `clip_and_aggregate(grads, C, σ)` | Per-sample clip + noise + aggregate |
| `clip_and_aggregate_microbatched(...)` | Memory-efficient microbatched variant |
| `check_model(model)` | Validate model compatibility |
| `patch_model_for_dp(model)` | Explicit SDPA backend selection (usually automatic) |

## LoRA Example

```python
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from private_mlx import make_private_loss, DPOptimizer
import mlx.nn as nn
from mlx.optimizers import Adam

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
model.freeze()
linear_to_lora_layers(model, num_layers=4,
    config={"rank": 8, "scale": 20.0, "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"]})

def lm_loss(model, x, y):
    logits = model(x[None, :])
    return nn.losses.cross_entropy(logits[0], y, reduction="mean")

dp_loss = make_private_loss(model, lm_loss)
optimizer = DPOptimizer(Adam(learning_rate=1e-4), l2_norm_clip=1.0,
                        noise_multiplier=1.0, target_delta=1e-5, num_samples=N)
```

## Personal Demo

Fine-tune on your own writing with DP, then generate completions:

```bash
cd experiments/mia
python demo.py --corpus my_writing.jsonl --noise-multiplier 1.0 --epochs 3
```

Corpus format: one `{"text": "..."}` per line. Not committed to the repo.

## Tests

```bash
pytest                   # unit tests (~22s, no downloads)
pytest -m mnist          # + MNIST integration (requires /tmp/mnist)
pytest -m lora           # + LoRA integration (downloads Qwen2.5-0.5B)
```

## License

MIT
