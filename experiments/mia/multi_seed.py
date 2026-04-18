"""Multi-seed MIA experiment + response-only scoring + canary ablation.

Usage:
    python multi_seed.py [--seeds 5] [--base-dir mia_run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from train import DEFAULT_CFG, SETTINGS, build_model, tokenize_corpus, train
from attack import run_attack

import mlx.core as mx


SETTINGS_LIST = ["non_dp", "dp_mid", "dp_strong"]


def run_multi_seed(base_dir: str, n_seeds: int = 5):
    corpus_path = os.path.join(base_dir, "canary_corpus.jsonl")
    split_path = os.path.join(base_dir, "split.json")

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line)["text"])

    with open(split_path) as f:
        split = json.load(f)

    member_texts = [corpus[i] for i in split["member_ids"]]

    seeds = [42, 123, 456, 789, 1337][:n_seeds]
    all_results = {s: [] for s in SETTINGS_LIST}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  SEED {seed} ({seed_idx+1}/{n_seeds})")
        print(f"{'='*60}")

        for setting in SETTINGS_LIST:
            run_dir = os.path.join(base_dir, "multi_seed", f"seed{seed}", setting)

            result_path = os.path.join(run_dir, "attack.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    metrics = json.load(f)
                print(f"  [{setting}] cached: AUC={metrics['roc_auc']}")
                all_results[setting].append(metrics)
                continue

            cfg = {**DEFAULT_CFG, **SETTINGS[setting]}
            cfg["seed"] = seed
            cfg["epochs"] = 5
            cfg["setting"] = setting

            _, tokenizer = build_model(cfg)
            train_x, train_y = tokenize_corpus(tokenizer, member_texts, cfg["seq_len"])
            mx.eval(train_x, train_y)

            print(f"  [{setting}] training seed={seed}...")
            model, meta = train(cfg, train_x, train_y, run_dir)
            del model
            mx.clear_cache()

            print(f"  [{setting}] scoring...")
            metrics = run_attack(run_dir, split_path, corpus_path, result_path)
            all_results[setting].append(metrics)

            eps_str = f"ε={metrics['epsilon']:.2f}" if metrics["epsilon"] else "ε=∞"
            print(f"  [{setting}] AUC={metrics['roc_auc']} bal_acc={metrics['balanced_accuracy']} {eps_str}")

    return all_results, seeds


def run_response_only(base_dir: str):
    """Re-score seed=42 runs with response-only loss (token position >= 6)."""
    print(f"\n{'='*60}")
    print("  RESPONSE-ONLY SCORING (response_start=6)")
    print(f"{'='*60}")

    corpus_path = os.path.join(base_dir, "canary_corpus.jsonl")
    split_path = os.path.join(base_dir, "split.json")
    results = {}

    for setting in SETTINGS_LIST:
        run_dir = os.path.join(base_dir, "runs", setting)
        output_path = os.path.join(base_dir, "results", f"{setting}_response_only.json")

        if os.path.exists(output_path):
            with open(output_path) as f:
                metrics = json.load(f)
        else:
            metrics = run_attack(
                run_dir, split_path, corpus_path, output_path,
                response_start=6,
            )

        results[setting] = metrics
        print(f"  [{setting}] AUC={metrics['roc_auc']} (response-only) vs full-seq baseline")

    return results


def run_canary_ablation(base_dir: str):
    """Generate unseen canaries and evaluate non-DP model on them.

    Tests whether the attack detects memorization vs template recognition.
    """
    print(f"\n{'='*60}")
    print("  CANARY ABLATION: memorization vs template recognition")
    print(f"{'='*60}")

    ablation_dir = os.path.join(base_dir, "canary_ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    rng = np.random.RandomState(9999)
    templates = [
        "The secret code for entry {i} is {code}. Remember it carefully.",
        "Patient {i} was diagnosed with condition {code} on January 15th.",
        "Account number {i} has balance {code} as of last quarter.",
        "Employee {i} rated performance score {code} in their review.",
        "Document {i} contains classified reference {code} on page three.",
    ]
    unseen = []
    for i in range(400, 600):
        t = templates[i % len(templates)]
        code = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), size=8))
        unseen.append({"text": t.format(i=i, code=code)})

    unseen_path = os.path.join(ablation_dir, "unseen_canaries.jsonl")
    with open(unseen_path, "w") as f:
        for item in unseen:
            f.write(json.dumps(item) + "\n")

    from attack import load_model_with_adapter, tokenize_corpus, score_losses
    from private_mlx._patch import ensure_attention_backend_for_per_sample_grads

    run_dir = os.path.join(base_dir, "runs", "non_dp")
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)

    model, tokenizer = load_model_with_adapter(meta, os.path.join(run_dir, "adapter.npz"))
    ensure_attention_backend_for_per_sample_grads(model, mode="auto", warn=False)

    unseen_texts = [item["text"] for item in unseen]
    ux, uy = tokenize_corpus(tokenizer, unseen_texts, meta.get("seq_len", 64))
    mx.eval(ux, uy)
    unseen_losses = score_losses(model, ux, uy)

    corpus_path = os.path.join(base_dir, "canary_corpus.jsonl")
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line)["text"])

    with open(os.path.join(base_dir, "split.json")) as f:
        split = json.load(f)

    all_texts = [corpus[i] for i in range(len(corpus))]
    ax, ay = tokenize_corpus(tokenizer, all_texts, meta.get("seq_len", 64))
    mx.eval(ax, ay)
    all_losses = score_losses(model, ax, ay)

    member_losses = [all_losses[i] for i in split["member_ids"]]
    nonmember_losses = [all_losses[i] for i in split["nonmember_ids"]]

    del model
    mx.clear_cache()

    result = {
        "member_loss_mean": round(float(np.mean(member_losses)), 4),
        "member_loss_std": round(float(np.std(member_losses)), 4),
        "nonmember_loss_mean": round(float(np.mean(nonmember_losses)), 4),
        "nonmember_loss_std": round(float(np.std(nonmember_losses)), 4),
        "unseen_canary_loss_mean": round(float(np.mean(unseen_losses)), 4),
        "unseen_canary_loss_std": round(float(np.std(unseen_losses)), 4),
        "n_unseen": len(unseen_losses),
    }

    with open(os.path.join(ablation_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Member loss:         {result['member_loss_mean']:.4f} ± {result['member_loss_std']:.4f}")
    print(f"  Nonmember loss:      {result['nonmember_loss_mean']:.4f} ± {result['nonmember_loss_std']:.4f}")
    print(f"  Unseen canary loss:  {result['unseen_canary_loss_mean']:.4f} ± {result['unseen_canary_loss_std']:.4f}")

    gap_member = abs(result["unseen_canary_loss_mean"] - result["member_loss_mean"])
    gap_nonmember = abs(result["unseen_canary_loss_mean"] - result["nonmember_loss_mean"])
    if gap_nonmember < gap_member:
        print("  -> Unseen canaries cluster with nonmembers: attack detects MEMORIZATION ✓")
    else:
        print("  -> Unseen canaries cluster with members: attack may detect TEMPLATE RECOGNITION ✗")

    return result


def print_summary(all_results: dict, seeds: list[int], resp_results: dict, ablation: dict):
    print(f"\n{'='*60}")
    print("  COMPREHENSIVE MIA RESULTS")
    print(f"{'='*60}")

    print(f"\n  Multi-seed variance ({len(seeds)} seeds: {seeds})")
    print(f"  {'Setting':<12} {'ε':>8} {'AUC':>14} {'Bal.Acc':>14} {'TPR@1%':>14}")
    print("  " + "-" * 64)
    for setting in SETTINGS_LIST:
        runs = all_results[setting]
        aucs = [r["roc_auc"] for r in runs]
        bals = [r["balanced_accuracy"] for r in runs]
        tprs = [r.get("tpr@fpr=0.01", 0.0) for r in runs]
        eps_vals = [r["epsilon"] for r in runs if r["epsilon"] is not None]
        eps_str = f"{np.mean(eps_vals):.2f}" if eps_vals else "∞"
        print(
            f"  {setting:<12} {eps_str:>8}"
            f" {np.mean(aucs):>6.4f}±{np.std(aucs):.4f}"
            f" {np.mean(bals):>6.4f}±{np.std(bals):.4f}"
            f" {np.mean(tprs):>6.4f}±{np.std(tprs):.4f}"
        )

    print(f"\n  Response-only scoring (token position ≥ 6)")
    print(f"  {'Setting':<12} {'Full AUC':>10} {'Resp AUC':>10} {'Delta':>8}")
    print("  " + "-" * 42)
    for setting in SETTINGS_LIST:
        seed42 = [r for r in all_results[setting] if True]  # just use first
        full_auc = seed42[0]["roc_auc"] if seed42 else 0
        resp_auc = resp_results.get(setting, {}).get("roc_auc", 0)
        delta = resp_auc - full_auc
        print(f"  {setting:<12} {full_auc:>10.4f} {resp_auc:>10.4f} {delta:>+8.4f}")

    print(f"\n  Canary ablation (non-DP model)")
    print(f"  Member loss:    {ablation['member_loss_mean']:.4f} ± {ablation['member_loss_std']:.4f}")
    print(f"  Nonmember loss: {ablation['nonmember_loss_mean']:.4f} ± {ablation['nonmember_loss_std']:.4f}")
    print(f"  Unseen loss:    {ablation['unseen_canary_loss_mean']:.4f} ± {ablation['unseen_canary_loss_std']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="mia_run")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    all_results, seeds = run_multi_seed(args.base_dir, args.seeds)
    resp_results = run_response_only(args.base_dir)
    ablation = run_canary_ablation(args.base_dir)

    print_summary(all_results, seeds, resp_results, ablation)

    output = {
        "seeds": seeds,
        "multi_seed": {s: all_results[s] for s in SETTINGS_LIST},
        "response_only": resp_results,
        "canary_ablation": ablation,
    }
    out_path = os.path.join(args.base_dir, "comprehensive_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
