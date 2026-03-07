#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals + McNemar's Significance Tests
File: tools/bootstrap_ci.py

Statistically validates the ablation results from TerrainFlood-UQ.

Two analyses:
  1. Bootstrap CI — chip-level resampling on IoU for a single model.
     Reports 95% CI so results can be stated as "IoU = X ± Y".

  2. McNemar's test — pixel-level paired comparison between two models.
     Reports p-value for "model B is significantly better than model A".
     Required for any claim like "C > B" or "D > C".

Why chip-level bootstrap (not pixel-level)?
  Pixels within the same chip are spatially correlated.
  Resampling individual pixels breaks spatial independence and
  underestimates variance. The correct unit of resampling is the chip.

Reference:
  Efron & Tibshirani (1993) — "An Introduction to the Bootstrap"
  McNemar (1947) — "Note on the sampling error of the difference between
                    correlated proportions or percentages"

Usage (single model — confidence intervals only):
  python tools/bootstrap_ci.py \\
      --checkpoint_a checkpoints/variant_D/best.pt \\
      --data_root    data/sen1floods11 \\
      --output_dir   results/stats \\
      --n_bootstrap  1000

Usage (two models — CI + McNemar pairwise test):
  python tools/bootstrap_ci.py \\
      --checkpoint_a checkpoints/variant_C/best.pt \\
      --checkpoint_b checkpoints/variant_D/best.pt \\
      --data_root    data/sen1floods11 \\
      --output_dir   results/stats \\
      --n_bootstrap  1000

Usage (full ablation — all 4 variants, all pairwise tests):
  python tools/bootstrap_ci.py \\
      --ablation \\
      --checkpoints_dir checkpoints \\
      --data_root       data/sen1floods11 \\
      --output_dir      results/stats \\
      --n_bootstrap     1000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2


# ─────────────────────────────────────────────────────────────
# Numeric-prefix imports (match the rest of the codebase)
# ─────────────────────────────────────────────────────────────

_root = Path(__file__).parent.parent


def _import_module(alias: str, file_path: str):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "model_bs"   not in sys.modules:
    _import_module("model_bs",   str(_root / "03_model.py"))
if "dataset_bs" not in sys.modules:
    _import_module("dataset_bs", str(_root / "02_dataset.py"))
if "unc_bs"     not in sys.modules:
    _import_module("unc_bs",     str(_root / "05_uncertainty.py"))

from model_bs   import build_model          # noqa: E402
from dataset_bs import get_dataloaders      # noqa: E402
from unc_bs     import mc_dropout_inference # noqa: E402


# ─────────────────────────────────────────────────────────────
# Per-chip prediction collection
# ─────────────────────────────────────────────────────────────

def collect_per_chip_stats(
    ckpt_path:   str,
    data_root:   str,
    device:      torch.device,
    split:       str = "test",
    T:           int = 1,
    num_workers: int = 2,
) -> list[dict]:
    """
    Runs inference on the requested split and returns per-chip
    confusion matrix entries needed for bootstrap resampling.

    Uses T=1 by default (deterministic) for speed. Set T=20 for Variant D
    to get variance-aware predictions (mean over MC passes).

    Args:
        ckpt_path:   path to best.pt
        data_root:   Sen1Floods11 root directory
        device:      torch device
        split:       "test" or "val"
        T:           MC Dropout passes (1 = deterministic)
        num_workers: dataloader workers

    Returns:
        list of dicts, one per chip, each containing:
            chip_id  : str
            event    : str
            tp       : int
            fp       : int
            fn       : int
            tn       : int
            n_valid  : int  — number of valid (non-ignore) pixels
            preds    : (H*W,) bool  — binary flood predictions (for McNemar)
            labels   : (H*W,) int   — ground truth (for McNemar)
    """
    ckpt    = torch.load(ckpt_path, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    print(f"  Loaded Variant {variant}  epoch={ckpt['epoch']}  "
          f"best_iou={ckpt.get('best_iou', 0.0):.4f}")

    _, val_loader, test_loader = get_dataloaders(
        data_root   = data_root,
        batch_size  = 1,
        num_workers = num_workers,
        pin_memory  = False,
    )
    loader = test_loader if split == "test" else val_loader

    # Run MC Dropout inference (T=1 is deterministic single pass)
    raw_results = mc_dropout_inference(model, loader, device, T=T)

    chips: list[dict] = []
    for r in raw_results:
        label     = r["label"]                           # (H, W) int
        mean_prob = r["mean_prob"]                       # (H, W) float32
        valid     = label != -1

        probs_v   = mean_prob[valid]
        labels_v  = label[valid]
        preds_v   = (probs_v > 0.5).astype(np.int32)
        labels_b  = (labels_v > 0).astype(np.int32)     # flood = 1 or 2 → 1

        tp = int(((preds_v == 1) & (labels_b == 1)).sum())
        fp = int(((preds_v == 1) & (labels_b == 0)).sum())
        fn = int(((preds_v == 0) & (labels_b == 1)).sum())
        tn = int(((preds_v == 0) & (labels_b == 0)).sum())

        chips.append({
            "chip_id":  r["chip_id"],
            "event":    r["event"],
            "tp":       tp,
            "fp":       fp,
            "fn":       fn,
            "tn":       tn,
            "n_valid":  int(valid.sum()),
            "preds":    preds_v.astype(bool),
            "labels":   labels_b.astype(bool),
        })

    return chips


# ─────────────────────────────────────────────────────────────
# Bootstrap confidence interval
# ─────────────────────────────────────────────────────────────

def bootstrap_iou_ci(
    chips:       list[dict],
    n_bootstrap: int   = 1000,
    ci:          float = 0.95,
    seed:        int   = 42,
) -> dict:
    """
    Chip-level bootstrap confidence interval on IoU.

    Resamples chips (with replacement) to respect spatial correlation.
    Pixel-level resampling is incorrect — pixels within a chip are NOT
    independent.

    Args:
        chips:       list of per-chip dicts (from collect_per_chip_stats)
        n_bootstrap: number of bootstrap resamples
        ci:          confidence level (default 0.95 → 95% CI)
        seed:        random seed for reproducibility

    Returns:
        dict with:
            iou_observed:  float  — IoU on original chips
            iou_mean:      float  — mean IoU across bootstrap samples
            iou_std:       float  — std of bootstrap IoUs
            ci_lower:      float  — lower bound of CI
            ci_upper:      float  — upper bound of CI
            ci_level:      float  — confidence level
            n_chips:       int
            n_bootstrap:   int
    """
    rng     = np.random.default_rng(seed)
    n_chips = len(chips)

    tp_arr = np.array([c["tp"] for c in chips])
    fp_arr = np.array([c["fp"] for c in chips])
    fn_arr = np.array([c["fn"] for c in chips])

    # Observed IoU
    iou_obs = tp_arr.sum() / max(tp_arr.sum() + fp_arr.sum() + fn_arr.sum(), 1)

    # Bootstrap
    iou_boot = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx    = rng.choice(n_chips, size=n_chips, replace=True)
        tp_b   = tp_arr[idx].sum()
        fp_b   = fp_arr[idx].sum()
        fn_b   = fn_arr[idx].sum()
        iou_boot[b] = tp_b / max(tp_b + fp_b + fn_b, 1)

    alpha  = 1.0 - ci
    lower  = float(np.percentile(iou_boot, 100 * alpha / 2))
    upper  = float(np.percentile(iou_boot, 100 * (1 - alpha / 2)))

    return {
        "iou_observed": round(float(iou_obs),         4),
        "iou_mean":     round(float(iou_boot.mean()),  4),
        "iou_std":      round(float(iou_boot.std()),   4),
        "ci_lower":     round(lower, 4),
        "ci_upper":     round(upper, 4),
        "ci_level":     ci,
        "n_chips":      n_chips,
        "n_bootstrap":  n_bootstrap,
    }


# ─────────────────────────────────────────────────────────────
# McNemar's test
# ─────────────────────────────────────────────────────────────

def mcnemar_test(
    chips_a: list[dict],
    chips_b: list[dict],
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    McNemar's test for statistical significance of difference between
    two flood segmentation models.

    H₀: models A and B make statistically identical errors.
    H₁: model B corrects significantly more pixels than it breaks.

    Pixel-level McNemar is correct here because we care about raw
    prediction disagreement, not chip-level aggregate.

    Notation:
        n01 = pixels where A is wrong, B is right  (B improves over A)
        n10 = pixels where A is right, B is wrong  (A is better)

    Statistic (with continuity correction):
        χ² = (|n01 − n10| − 1)² / (n01 + n10)
    Under H₀: χ² ~ chi-squared(df=1)

    Args:
        chips_a: per-chip stats from collect_per_chip_stats() for model A
        chips_b: per-chip stats from collect_per_chip_stats() for model B
        label_a: display name for model A
        label_b: display name for model B

    Returns:
        dict with statistic, p_value, n01, n10, interpretation
    """
    # Align chips by chip_id
    id_to_a = {c["chip_id"]: c for c in chips_a}
    id_to_b = {c["chip_id"]: c for c in chips_b}
    common  = sorted(set(id_to_a) & set(id_to_b))

    if not common:
        return {"error": "no common chips found between checkpoints"}

    n01 = 0   # A wrong, B right
    n10 = 0   # A right, B wrong

    for cid in common:
        ca = id_to_a[cid]
        cb = id_to_b[cid]

        # Align predictions using TP/FP/FN/TN counts at chip level
        # (we can't recover pixel-by-pixel ordering from aggregate stats,
        #  but we can compute the paired pixel count from raw preds arrays)
        preds_a = ca["preds"].astype(bool)   # (N,) binary prediction
        preds_b = cb["preds"].astype(bool)
        labels  = ca["labels"].astype(bool)  # same for both

        # Truncate to the shorter length if they differ (safety)
        n = min(len(preds_a), len(preds_b), len(labels))
        preds_a = preds_a[:n]
        preds_b = preds_b[:n]
        labels  = labels[:n]

        a_wrong = preds_a != labels
        b_wrong = preds_b != labels

        n01 += int(( a_wrong & ~b_wrong).sum())
        n10 += int((~a_wrong &  b_wrong).sum())

    n_discordant = n01 + n10
    if n_discordant == 0:
        return {
            "label_a":       label_a,
            "label_b":       label_b,
            "n01_A_wrong_B_right": 0,
            "n10_A_right_B_wrong": 0,
            "net_improvement":     0,
            "statistic":     0.0,
            "p_value":       1.0,
            "significant":   False,
            "interpretation": "identical predictions — no difference",
            "n_common_chips": len(common),
        }

    # McNemar with continuity correction (Yates 1934)
    stat    = (abs(n01 - n10) - 1.0) ** 2 / n_discordant
    p_value = float(1.0 - chi2.cdf(stat, df=1))

    if p_value < 0.001:
        stars = "***"
        verdict = "highly significant"
    elif p_value < 0.01:
        stars = "**"
        verdict = "significant"
    elif p_value < 0.05:
        stars = "*"
        verdict = "significant"
    else:
        stars = "n.s."
        verdict = "NOT significant"

    direction = f"{label_b} > {label_a}" if n01 > n10 else f"{label_a} > {label_b}"

    return {
        "label_a":                label_a,
        "label_b":                label_b,
        "n01_A_wrong_B_right":    n01,
        "n10_A_right_B_wrong":    n10,
        "net_improvement_B":      n01 - n10,
        "statistic":              round(float(stat),    4),
        "p_value":                round(p_value,        6),
        "stars":                  stars,
        "significant":            p_value < 0.05,
        "direction":              direction,
        "interpretation":         f"{verdict} {stars} — {direction}",
        "n_common_chips":         len(common),
    }


# ─────────────────────────────────────────────────────────────
# Bootstrap distribution figure
# ─────────────────────────────────────────────────────────────

def plot_bootstrap_distributions(
    results:  dict,
    out_path: Path,
) -> None:
    """
    Bar chart of observed IoU with 95% CI error bars for each variant.
    """
    variants = sorted(results.keys())
    ious     = [results[v]["ci"]["iou_observed"] for v in variants]
    ci_lower = [results[v]["ci"]["iou_observed"] - results[v]["ci"]["ci_lower"]
                for v in variants]
    ci_upper = [results[v]["ci"]["ci_upper"] - results[v]["ci"]["iou_observed"]
                for v in variants]

    colors = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52"}

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars = ax.bar(
        variants, ious,
        yerr=[ci_lower, ci_upper],
        capsize=6, color=[colors.get(v, "grey") for v in variants],
        alpha=0.85, edgecolor="white", linewidth=1.2,
        error_kw={"linewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
    )

    # Add IoU value labels
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2,
                iou + max(ci_upper) + 0.005,
                f"{iou:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Variant",  fontsize=11)
    ax.set_ylabel("IoU",      fontsize=11)
    ax.set_title("Bolivia OOD IoU with 95% Bootstrap CI (chip-level resampling)",
                 fontsize=11)
    ax.set_ylim(0, min(1.0, max(ious) + max(ci_upper) + 0.05))
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Figure → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main runners
# ─────────────────────────────────────────────────────────────

def run_single(args: argparse.Namespace, device: torch.device) -> None:
    """Single checkpoint: bootstrap CI only."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCollecting predictions: {args.checkpoint_a}")
    chips = collect_per_chip_stats(
        ckpt_path   = args.checkpoint_a,
        data_root   = args.data_root,
        device      = device,
        split       = args.split,
        T           = args.T,
        num_workers = args.num_workers,
    )

    print(f"\nBootstrap CI (n={args.n_bootstrap})...")
    ci = bootstrap_iou_ci(chips, n_bootstrap=args.n_bootstrap, seed=args.seed)

    print(f"\n{'='*50}")
    print(f"  IoU observed : {ci['iou_observed']:.4f}")
    print(f"  95% CI       : [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  ± (half-width): ±{(ci['ci_upper'] - ci['ci_lower']) / 2:.4f}")
    print(f"  Bootstrap std: {ci['iou_std']:.4f}")
    print(f"  n_chips      : {ci['n_chips']}")
    print(f"{'='*50}")

    out_json = out_dir / "bootstrap_ci_single.json"
    out_json.write_text(json.dumps(ci, indent=2))
    print(f"JSON → {out_json}")


def run_pair(args: argparse.Namespace, device: torch.device) -> None:
    """Two checkpoints: CI for both + McNemar's test."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCollecting predictions: A = {args.checkpoint_a}")
    chips_a = collect_per_chip_stats(
        ckpt_path   = args.checkpoint_a,
        data_root   = args.data_root,
        device      = device,
        split       = args.split,
        T           = args.T,
        num_workers = args.num_workers,
    )

    print(f"\nCollecting predictions: B = {args.checkpoint_b}")
    chips_b = collect_per_chip_stats(
        ckpt_path   = args.checkpoint_b,
        data_root   = args.data_root,
        device      = device,
        split       = args.split,
        T           = args.T,
        num_workers = args.num_workers,
    )

    print("\nBootstrap CI for A...")
    ci_a = bootstrap_iou_ci(chips_a, n_bootstrap=args.n_bootstrap, seed=args.seed)
    print("Bootstrap CI for B...")
    ci_b = bootstrap_iou_ci(chips_b, n_bootstrap=args.n_bootstrap, seed=args.seed)

    print("\nMcNemar's test...")
    mcn = mcnemar_test(chips_a, chips_b,
                       label_a=args.label_a, label_b=args.label_b)

    # Summary
    print(f"\n{'='*55}")
    print(f"  {args.label_a} IoU = {ci_a['iou_observed']:.4f}  "
          f"95%CI [{ci_a['ci_lower']:.4f}, {ci_a['ci_upper']:.4f}]")
    print(f"  {args.label_b} IoU = {ci_b['iou_observed']:.4f}  "
          f"95%CI [{ci_b['ci_lower']:.4f}, {ci_b['ci_upper']:.4f}]")
    print(f"\n  McNemar: χ²={mcn['statistic']:.4f}  p={mcn['p_value']:.6f}  "
          f"{mcn['stars']}")
    print(f"  {mcn['interpretation']}")
    print(f"{'='*55}")

    payload = {
        f"ci_{args.label_a}": ci_a,
        f"ci_{args.label_b}": ci_b,
        "mcnemar":            mcn,
    }
    out_json = out_dir / f"stats_{args.label_a}_vs_{args.label_b}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"JSON → {out_json}")


def run_ablation(args: argparse.Namespace, device: torch.device) -> None:
    """
    Full ablation: bootstrap CI for all variants + all pairwise McNemar tests.
    Expects checkpoints at:
        checkpoints_dir/variant_A/best.pt  (required)
        checkpoints_dir/variant_B/best.pt  (required)
        checkpoints_dir/variant_C/best.pt  (required)
        checkpoints_dir/variant_D/best.pt  (required)
        checkpoints_dir/variant_E/best.pt  (optional — skipped if absent)
    """
    out_dir       = Path(args.output_dir)
    ckpt_dir      = Path(args.checkpoints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["A", "B", "C", "D", "E"]
    variant_chips: dict[str, list[dict]] = {}
    results:       dict[str, dict]       = {}

    # Collect predictions for each variant
    for v in variants:
        ckpt_path = ckpt_dir / f"variant_{v}" / "best.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path} not found")
            continue
        print(f"\n── Variant {v} ──")
        chips = collect_per_chip_stats(
            ckpt_path   = str(ckpt_path),
            data_root   = args.data_root,
            device      = device,
            split       = args.split,
            T           = args.T,
            num_workers = args.num_workers,
        )
        variant_chips[v] = chips

        ci = bootstrap_iou_ci(chips, n_bootstrap=args.n_bootstrap, seed=args.seed)
        results[v] = {"ci": ci}
        print(f"  IoU = {ci['iou_observed']:.4f}  "
              f"95%CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Pairwise McNemar tests
    pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C"), ("A", "D")]
    mcnemar_results: list[dict] = []

    print(f"\n{'─'*55}")
    print("McNemar pairwise significance tests:")
    print(f"{'─'*55}")

    for (va, vb) in pairs:
        if va not in variant_chips or vb not in variant_chips:
            continue
        mcn = mcnemar_test(
            variant_chips[va], variant_chips[vb],
            label_a=f"Variant_{va}", label_b=f"Variant_{vb}",
        )
        mcnemar_results.append(mcn)
        print(f"  {va} vs {vb}: χ²={mcn['statistic']:7.2f}  "
              f"p={mcn['p_value']:.4f}  {mcn['stars']:4s}  "
              f"{mcn['interpretation']}")

    # Save JSON
    payload = {
        "variants":  results,
        "mcnemar":   mcnemar_results,
        "n_bootstrap": args.n_bootstrap,
        "split":     args.split,
    }
    out_json = out_dir / "ablation_stats.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON → {out_json}")

    # Figure
    if len(results) >= 2:
        out_fig = out_dir / "bootstrap_ci_ablation.png"
        plot_bootstrap_distributions(
            {v: results[v] for v in results},
            out_fig,
        )

    # Print thesis-ready table
    print(f"\n{'='*65}")
    print(f"  {'Variant':<10} {'IoU':>8} {'95% CI Lower':>14} {'95% CI Upper':>14}")
    print(f"  {'─'*10} {'─'*8} {'─'*14} {'─'*14}")
    for v, r in results.items():
        ci = r["ci"]
        print(f"  {v:<10} {ci['iou_observed']:>8.4f} "
              f"{ci['ci_lower']:>14.4f} {ci['ci_upper']:>14.4f}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap CI + McNemar significance tests for ablation"
    )

    # Mode selection
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--ablation",        action="store_true",
                      help="Full ablation: all 4 variants + all pairwise tests")

    # Checkpoint inputs
    p.add_argument("--checkpoint_a",    type=str, default=None,
                   help="Checkpoint for model A (or single model for CI only)")
    p.add_argument("--checkpoint_b",    type=str, default=None,
                   help="Checkpoint for model B (optional — enables McNemar)")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                   help="Root directory containing variant_A/.../best.pt (--ablation mode)")
    p.add_argument("--label_a",         type=str, default="A",
                   help="Display label for model A")
    p.add_argument("--label_b",         type=str, default="B",
                   help="Display label for model B")

    # Shared options
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/stats")
    p.add_argument("--split",        type=str, default="test",
                   choices=["test", "val"],
                   help="test=Bolivia OOD (15 chips), val=Paraguay (67 chips)")
    p.add_argument("--T",            type=int, default=1,
                   help="MC Dropout passes (1=deterministic, 20 for Variant D)")
    p.add_argument("--n_bootstrap",  type=int, default=1000,
                   help="Number of bootstrap resamples (≥1000 recommended)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_workers",  type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.ablation:
        run_ablation(args, device)
    elif args.checkpoint_b is not None:
        run_pair(args, device)
    elif args.checkpoint_a is not None:
        run_single(args, device)
    else:
        print("ERROR: provide --checkpoint_a, or --checkpoint_a + --checkpoint_b, "
              "or --ablation + --checkpoints_dir")
        raise SystemExit(1)
