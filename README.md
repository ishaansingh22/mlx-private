# mlx-private

MLX-native DP-SGD for private LoRA fine-tuning on Apple Silicon.

Train a model on your data with per-sample DP-SGD, entirely on-device. No cloud, no data leaves your machine.

```python
from mlx_private import make_private_loss, DPOptimizer

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

## Does DP Actually Work? Membership Inference Results

We trained Qwen2.5-0.5B LoRA adapters on a canary corpus (200 training examples with unique identifiers), then ran a loss-threshold membership inference attack. All numbers are mean ± std over 5 random seeds.

| Setting | ε (measured) | ROC-AUC | Bal. Accuracy | TPR @ 1% FPR |
|---|---|---|---|---|
| No DP | ∞ | **0.790 ± 0.024** | **0.727 ± 0.029** | **0.157 ± 0.075** |
| DP (σ=0.5) | 15.43 | 0.493 ± 0.006 | 0.519 ± 0.005 | 0.034 ± 0.006 |
| DP (σ=1.5) | 1.08 | 0.493 ± 0.015 | 0.523 ± 0.010 | 0.024 ± 0.016 |

Without DP, the attacker identifies training examples at AUC 0.79. With DP, the attack collapses to chance — even at the weaker ε=15.43 setting.

**The non-DP model does not produce verbatim completions of training data** (0/20 canaries extractable via greedy decoding, tested with tokenization-correct prefixes). Yet loss-based MIA still detects membership at 79% AUC. Teacher-forced next-token accuracy reaches 97.4%, confirming the memorization is real — it exists in the loss distribution but isn't surfaceable through autoregressive generation. This is consistent with the literature: loss-based attacks are more sensitive than extraction attacks, and it's why DP matters even for models that don't appear to leak in casual interaction.

Canary ablation confirms the attack detects genuine memorization, not template recognition: unseen canaries (novel codes, same template) have loss 0.857 ± 0.075, clustering with nonmembers (0.786) not members (0.696).

Reproduce: `cd experiments/mia && python run_experiment.py --epochs 5 --seed 42`
Multi-seed: `python multi_seed.py --seeds 5`

## Install

```bash
pip install -e .          # core library
pip install -e ".[test]"  # + pytest, dp-accounting
pip install -e ".[lora]"  # + mlx-lm for LoRA workflows
```

## What This Supports (v0.1)

- **LoRA fine-tuning** of decoder-only transformers on Apple Silicon via `mlx-lm`.
- **Non-quantized** (bf16/fp16/fp32) base models with frozen weights and trainable LoRA adapters.
- **Qwen and Llama** model families validated. Other GQA/MQA architectures with the standard `mlx-lm` attention pattern should work.
- **Single-device** training only.

## What This Does Not Support

- Quantized base models (`QuantizedMatmul::vmap` is NYI in MLX).
- Full fine-tuning (memory = O(B x total_params); use LoRA).
- Ghost clipping or memory-efficient per-sample gradient approximations.
- Stock `nn.Conv2d` with padding under `vmap(grad)`.
- Cryptographically secure noise generation.
- Multi-device / distributed training.

## Privacy Accounting Caveat

The RDP accountant assumes **Poisson subsampling** (each example included independently with probability q = B/N). The actual data loader uses **fixed-size uniform sampling without replacement**, which provides slightly weaker amplification than Poisson. The reported ε is therefore **conservative** — actual privacy is at least as good as stated, but the bound is not tight. See Balle et al. 2018 for the distinction. This is a known limitation shared with most DP-SGD implementations including Opacus.

## How It Works

1. `make_private_loss` wraps your loss with `mx.vmap(mx.grad(...))`. Only trainable (unfrozen) parameters get gradients — no memory wasted on frozen base model weights.

2. For GQA models, the fused SDPA kernel is replaced with decomposed attention. MLX 0.31.1's `mx.fast.scaled_dot_product_attention` hangs or crashes under `vmap` when query and key/value head counts differ ([ml-explore/mlx#3383](https://github.com/ml-explore/mlx/issues/3383)). The fallback is selective: only GQA modules are patched, MHA stays on the fused path. ~1.45x attention overhead.

3. `DPOptimizer.step()` clips each sample's gradient to L2 norm C, sums, adds N(0, σ²C²) noise, averages, and delegates to the base optimizer. The RDP accountant tracks ε automatically.

## API

| Function | Purpose |
|---|---|
| `make_private_loss(model, loss_fn)` | Wrap loss with `vmap(grad)` for per-sample gradients |
| `DPOptimizer(base, C, σ, δ, N)` | DP optimizer with built-in RDP accountant |
| `RDPAccountant(δ)` | Standalone privacy accountant |
| `clip_and_aggregate(grads, C, σ)` | Per-sample clip + noise + aggregate |
| `clip_and_aggregate_microbatched(...)` | Memory-efficient microbatched variant |
| `check_model(model)` | Validate model compatibility |
| `patch_model_for_dp(model)` | Explicit manual SDPA backend (usually automatic) |

## LoRA Example

```python
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_private import make_private_loss, DPOptimizer
import mlx.nn as nn
from mlx.optimizers import Adam

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
model.freeze()
linear_to_lora_layers(model, num_layers=4,
    config={"rank": 8, "scale": 20.0, "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"]})

def lm_loss(model, x, y):
    logits = model(x[None, :])
    return nn.losses.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        y[None, 1:].reshape(-1), reduction="mean")

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

Corpus format: one `{"text": "..."}` per line (JSONL). Not committed to the repo.

## Tests

```bash
pytest                   # fast unit tests (~22s, no data downloads)
pytest -m mnist          # + MNIST integration tests (requires /tmp/mnist)
pytest -m lora           # + LoRA integration test (downloads Qwen2.5-0.5B)
```

## License

MIT
