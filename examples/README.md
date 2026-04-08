# Examples

Two self-contained scripts that reproduce the README results. Both run end-to-end with one command on a 16 GB Mac.

## Canary frontier

Worst-case stress test: synthetic corpus with planted unique identifiers.

```bash
python3 examples/canary_frontier.py
```

**Runtime:** ~8 min on M1 Pro 16 GB.

**Expected output:** Non-DP AUC ~0.83, both DP settings collapse to ~0.50.

## PubMedQA

Privacy validation on real medical text (PubMedQA, 500 train / 500 test).

```bash
python3 examples/pubmedqa_dp.py
```

**Runtime:** ~25 min on M1 Pro 16 GB.

**Expected output:** Non-DP MIA AUC ~0.75–0.80, DP settings collapse to ~0.50. The fine-tune does not learn the classification task at this LoRA rank / data scale (accuracy ≈ majority baseline); the experiment validates the privacy mechanism on real text, not the model's task utility.

## Requirements

Both scripts require the `lora` extras:

```bash
pip install -e ".[lora]"    # mlx-lm, numpy
```

PubMedQA data is downloaded from HuggingFace on first run (no extra dependencies).
