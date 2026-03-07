#!/usr/bin/env python3
"""
Uncertainty–Error Correlation Analysis
File: tools/uncertainty_error_correlation.py

Validates that MC Dropout uncertainty is meaningful by showing that high-variance
pixels are also high-error pixels. This is the key claim that makes UQ useful:
the model knows when it does not know.

Analysis:
  1. Run MC Dropout (T passes) on the test set with Variant D.
  2. For each valid pixel, record:
       - predictive variance  (MC variance across T sigmoid outputs)
       - whether the prediction is correct  (max-prob class == label)
  3. Bin pixels by uncertainty percentile (0–100 in steps of 5).
  4. In each bin, compute the error rate (fraction of wrong predictions).
  5. A good UQ model: error rate should *increase monotonically* with uncertainty.
  6. Report Pearson/Spearman correlation between bin uncertainty and bin error rate.
  7. Save figure + JSON summary.

Outputs (in --output_dir, default results/uncertainty_D):
  - error_correlation.json   — bin table + Pearson r + Spearman ρ
  - error_vs_uncertainty.png — publication-quality scatter + regression line

Usage:
  python tools/uncertainty_error_correlation.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/uncertainty_D \\
      --T          20

  # Quick sanity check (T=5):
  python tools/uncertainty_error_correlation.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --T          5
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
from scipy import stats


# ─────────────────────────────────────────────────────────────
# Numeric-prefix imports
# ─────────────────────────────────────────────────────────────

_root = Path(__file__).parent.parent


def _import_module(alias: str, file_path: str):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "model_uec" not in sys.modules:
    _import_module("model_uec",   str(_root / "03_model.py"))
if "dataset_uec" not in sys.modules:
    _import_module("dataset_uec", str(_root / "02_dataset.py"))

from model_uec   import build_model      # noqa: E402
from dataset_uec import get_dataloaders  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Inference with MC Dropout
# ─────────────────────────────────────────────────────────────

def collect_pixels(
    ckpt_path:   str,
    data_root:   str,
    device:      torch.device,
    split:       str = "test",
    T:           int = 20,
    num_workers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MC Dropout inference and return flat arrays of per-pixel variance
    and per-pixel error indicator (1 = wrong, 0 = correct).

    Only valid pixels (label != -1) are included.

    Args:
        ckpt_path:   path to best.pt (Variant D)
        data_root:   Sen1Floods11 root directory
        device:      torch device
        split:       "test" or "val"
        T:           MC Dropout forward passes
        num_workers: dataloader workers

    Returns:
        variances: (N,) float32 — predictive variance per pixel
        errors:    (N,) bool    — True where prediction is wrong
    """
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    variant = config.get("variant", "D")

    if variant != "D":
        print(f"[WARN] Checkpoint variant={variant} — MC Dropout UQ is designed for Variant D. "
              "Proceeding anyway but results may not be meaningful.")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    model.enable_dropout()

    print(f"Loaded Variant {variant}  epoch={ckpt['epoch']}  "
          f"best_iou={ckpt.get('best_iou', 0.0):.4f}")
    print(f"Split: {split}  T={T}\n")

    _, val_loader, test_loader = get_dataloaders(
        data_root   = data_root,
        batch_size  = 1,
        num_workers = num_workers,
        patch_size  = None,
        pin_memory  = False,
    )
    loader = test_loader if split == "test" else val_loader

    all_var:   list[np.ndarray] = []
    all_error: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device)   # (1, 6, H, W)
            labels  = batch["label"].numpy()[0]    # (H, W) int
            chip_id = batch["chip_id"][0]

            # T stochastic passes → (T, H, W) probability maps
            probs_list = []
            for _ in range(T):
                logits = model(images)                              # (1, 1, H, W)
                prob   = torch.sigmoid(logits.squeeze()).cpu().numpy()  # (H, W)
                probs_list.append(prob)

            probs = np.stack(probs_list, axis=0)        # (T, H, W)
            mean_p = probs.mean(axis=0)                  # (H, W)
            var_p  = probs.var(axis=0)                   # (H, W)

            # Flood label: 1 or 2 → flood; 0 → no flood; -1 → ignore
            valid = labels != -1
            target_flood = ((labels == 1) | (labels == 2))[valid]
            pred_flood   = (mean_p > 0.5)[valid]
            variance_v   = var_p[valid]

            error = (pred_flood != target_flood)

            all_var.append(variance_v.astype(np.float32))
            all_error.append(error.astype(np.float32))

            n_px = valid.sum()
            err_rate = error.mean()
            mean_var = variance_v.mean()
            print(f"  {chip_id:<35}  n_px={n_px:7d}  error_rate={err_rate:.3f}  "
                  f"mean_var={mean_var:.4f}")

    return (
        np.concatenate(all_var),
        np.concatenate(all_error).astype(bool),
    )


# ─────────────────────────────────────────────────────────────
# Binned error-rate analysis
# ─────────────────────────────────────────────────────────────

def bin_error_by_uncertainty(
    variances: np.ndarray,
    errors:    np.ndarray,
    n_bins:    int = 20,
) -> dict:
    """
    Bin pixels by uncertainty percentile and compute error rate per bin.

    Args:
        variances: (N,) per-pixel predictive variance
        errors:    (N,) bool, True = wrong prediction
        n_bins:    number of equal-frequency bins (default 20 → 5% each)

    Returns:
        dict with keys:
          bin_centers     — list[float] median variance in each bin
          bin_error_rates — list[float] error rate in each bin
          bin_counts      — list[int]   pixel count per bin
          pearson_r       — float Pearson correlation
          pearson_p       — float p-value
          spearman_rho    — float Spearman rank correlation
          spearman_p      — float p-value
    """
    # Equal-frequency binning via percentile edges
    percentile_edges = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(variances, percentile_edges)
    bin_edges[-1] += 1e-12  # include max value in last bin
    bin_indices = np.digitize(variances, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    centers:    list[float] = []
    error_rates: list[float] = []
    counts:     list[int]   = []

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        centers.append(float(variances[mask].mean()))
        error_rates.append(float(errors[mask].mean()))
        counts.append(int(mask.sum()))

    centers_arr    = np.array(centers)
    error_rate_arr = np.array(error_rates)

    r, p_r   = stats.pearsonr(centers_arr, error_rate_arr)
    rho, p_s = stats.spearmanr(centers_arr, error_rate_arr)

    return {
        "n_bins":           n_bins,
        "bin_centers":      [round(float(c), 6) for c in centers],
        "bin_error_rates":  [round(float(e), 4) for e in error_rates],
        "bin_counts":       counts,
        "pearson_r":        round(float(r),   4),
        "pearson_p":        round(float(p_r), 6),
        "spearman_rho":     round(float(rho), 4),
        "spearman_p":       round(float(p_s), 6),
        "total_pixels":     int(len(variances)),
        "overall_error_rate": round(float(errors.mean()), 4),
        "overall_mean_var":   round(float(variances.mean()), 6),
    }


# ─────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────

def plot_error_correlation(
    result:   dict,
    out_path: Path,
    variant:  str = "D",
    split:    str = "test",
    T:        int = 20,
) -> None:
    """
    Plot error rate vs. uncertainty bin with regression line.

    Args:
        result:   output of bin_error_by_uncertainty()
        out_path: save path (.png)
        variant:  model variant label
        split:    data split label
        T:        number of MC Dropout passes
    """
    centers    = np.array(result["bin_centers"])
    err_rates  = np.array(result["bin_error_rates"])
    counts     = np.array(result["bin_counts"])

    # Regression line
    slope, intercept, _, _, _ = stats.linregress(centers, err_rates)
    x_line = np.linspace(centers.min(), centers.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)

    # Scatter: size proportional to bin count
    sizes = 30 + 200 * (counts / counts.max())
    sc = ax.scatter(
        centers, err_rates,
        s=sizes, c=centers,
        cmap="plasma", alpha=0.85, zorder=3,
        label="Uncertainty bins (size ∝ pixel count)"
    )
    ax.plot(x_line, y_line, "r--", linewidth=1.5, alpha=0.8, label="Linear fit")

    plt.colorbar(sc, ax=ax, label="Mean predictive variance")

    ax.set_xlabel("Mean predictive variance (per bin)", fontsize=11)
    ax.set_ylabel("Error rate (fraction wrong)", fontsize=11)
    ax.set_title(
        f"Uncertainty–Error Correlation — Variant {variant} | {split} | T={T}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    # Annotation
    r   = result["pearson_r"]
    rho = result["spearman_rho"]
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r:.3f}  (p={result['pearson_p']:.4f})\n"
        f"Spearman ρ = {rho:.3f}  (p={result['spearman_p']:.4f})\n"
        f"Total pixels: {result['total_pixels']:,}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    ax.legend(loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Figure → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run_uncertainty_error_correlation(
    ckpt_path:   str,
    data_root:   str,
    output_dir:  str,
    device:      torch.device,
    split:       str = "test",
    T:           int = 20,
    n_bins:      int = 20,
    num_workers: int = 2,
) -> None:
    """
    Full pipeline: MC Dropout inference → binned analysis → figure + JSON.

    Args:
        ckpt_path:   path to best.pt (Variant D)
        data_root:   Sen1Floods11 root
        output_dir:  output directory
        device:      torch device
        split:       "test" or "val"
        T:           MC Dropout passes
        n_bins:      number of uncertainty bins (default 20 = 5% each)
        num_workers: dataloader workers
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output:     {out_dir}\n")

    # Step 1: Collect per-pixel variance and error
    variances, errors = collect_pixels(
        ckpt_path   = ckpt_path,
        data_root   = data_root,
        device      = device,
        split       = split,
        T           = T,
        num_workers = num_workers,
    )

    print(f"\nCollected {len(variances):,} valid pixels")
    print(f"Overall error rate: {errors.mean():.3f}")
    print(f"Mean variance:      {variances.mean():.5f}")
    print(f"Max variance:       {variances.max():.5f}\n")

    # Step 2: Binned analysis
    result = bin_error_by_uncertainty(variances, errors, n_bins=n_bins)

    r   = result["pearson_r"]
    rho = result["spearman_rho"]
    print(f"Pearson r  = {r:.4f}  (p={result['pearson_p']:.6f})")
    print(f"Spearman ρ = {rho:.4f}  (p={result['spearman_p']:.6f})\n")

    # Interpret
    if r > 0.7:
        interp = "Strong positive correlation — uncertainty reliably identifies hard pixels."
    elif r > 0.4:
        interp = "Moderate positive correlation — uncertainty is a useful error proxy."
    elif r > 0.0:
        interp = "Weak positive correlation — some UQ signal present."
    else:
        interp = "No/negative correlation — uncertainty does not align with errors."

    print(f"Interpretation: {interp}\n")

    # Load checkpoint info for metadata
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    variant = ckpt.get("config", {}).get("variant", "D")

    # Step 3: Save JSON
    out_json = out_dir / "error_correlation.json"
    payload  = {
        "checkpoint":  ckpt_path,
        "variant":     variant,
        "split":       split,
        "T":           T,
        "n_bins":      n_bins,
        "interpretation": interp,
        **result,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"JSON → {out_json}")

    # Step 4: Figure
    out_png = out_dir / "error_vs_uncertainty.png"
    plot_error_correlation(
        result   = result,
        out_path = out_png,
        variant  = variant,
        split    = split,
        T        = T,
    )

    # Console summary table
    print(f"\n{'Uncertainty bin':>18} {'Mean var':>12} {'Error rate':>12} {'N pixels':>10}")
    print("-" * 56)
    for i, (c, e, n) in enumerate(
        zip(result["bin_centers"], result["bin_error_rates"], result["bin_counts"])
    ):
        pct_lo = (i * 100) // n_bins
        pct_hi = ((i + 1) * 100) // n_bins
        print(f"  p{pct_lo:02d}–p{pct_hi:02d}            {c:12.5f} {e:12.4f} {n:10d}")

    print(f"\nPearson r={r:.4f}  Spearman ρ={rho:.4f}")
    print(f"Summary → {out_json}")
    print(f"Figure  → {out_png}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate that MC Dropout uncertainty correlates with prediction errors"
    )
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to best.pt (Variant D)")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/uncertainty_D")
    p.add_argument("--split",        type=str, default="test",
                   choices=["test", "val"],
                   help="test = Bolivia OOD (15 chips), val = Paraguay (67 chips)")
    p.add_argument("--T",            type=int, default=20,
                   help="MC Dropout passes (>=10 recommended for stable variance estimates)")
    p.add_argument("--n_bins",       type=int, default=20,
                   help="Number of equal-frequency uncertainty bins")
    p.add_argument("--num_workers",  type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_uncertainty_error_correlation(
        ckpt_path   = args.checkpoint,
        data_root   = args.data_root,
        output_dir  = args.output_dir,
        device      = device,
        split       = args.split,
        T           = args.T,
        n_bins      = args.n_bins,
        num_workers = args.num_workers,
    )
