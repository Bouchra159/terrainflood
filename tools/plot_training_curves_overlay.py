#!/usr/bin/env python3
"""
Multi-variant training curve overlay figure.
File: tools/plot_training_curves_overlay.py

Loads the per-variant CSV files produced by tools/export_tb_curves.py and
generates a single publication-quality figure with all 4 variants overlaid.

Expected CSV format (produced by export_tb_curves.py after the column-name fix):
    epoch, train_loss, val_loss, train_iou, val_iou [, lr]

Usage:
    # After running export_tb_curves.py for each variant:
    python tools/plot_training_curves_overlay.py \\
        --curves_dir results/curves \\
        --out_dir    results/paper_figures \\
        --variant_names A B C D

    # Custom CSV paths:
    python tools/plot_training_curves_overlay.py \\
        --csv_A results/curves/variant_A_scalars.csv \\
        --csv_B results/curves/variant_B_scalars.csv \\
        --csv_C results/curves/variant_C_scalars.csv \\
        --csv_D results/curves/variant_D_scalars.csv \\
        --out_dir results/paper_figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("Missing dependency: pandas. pip install pandas") from e


# ─────────────────────────────────────────────────────────────
# Publication style (must stay consistent with plots.py)
# ─────────────────────────────────────────────────────────────

def _set_pub_style() -> None:
    plt.rcParams.update({
        "figure.dpi":         160,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "legend.fontsize":    8.5,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "lines.linewidth":    1.5,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.20,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "grid.color":         "#888888",
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.7",
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "mathtext.fontset":   "stix",
    })


# Okabe–Ito colour-blind-safe palette  (4 variants)
_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
_MARKERS = ["o", "s", "^", "D"]
_LABELS  = [
    "Variant A (SAR only)",
    "Variant B (HAND band)",
    "Variant C (HAND gate)",
    "Variant D (HAND gate + MC Dropout)",
]


# ─────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load variant CSV and return DataFrame, or None if missing/empty."""
    if path is None or not path.exists():
        print(f"  [SKIP] Not found: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  [SKIP] Empty CSV: {path}")
        return None
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy: substring match (case-insensitive)
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def _extract_series(df: pd.DataFrame) -> Tuple[
    Optional[np.ndarray],   # epochs
    Optional[np.ndarray],   # val_iou
    Optional[np.ndarray],   # val_loss
]:
    """Extract epochs, val_iou, val_loss arrays from a variant CSV."""
    epoch_col    = _find_col(df, ["epoch", "Epoch", "step", "Step"])
    val_iou_col  = _find_col(df, ["val_iou",  "val/iou",  "iou_val",  "IoU/val"])
    val_loss_col = _find_col(df, ["val_loss", "val/loss", "loss_val", "Loss/val"])

    if epoch_col is None:
        print("  [WARN] No epoch column found — skipping this variant.")
        return None, None, None

    epochs = df[epoch_col].values.astype(float)
    val_iou  = df[val_iou_col].values.astype(float)  if val_iou_col  else None
    val_loss = df[val_loss_col].values.astype(float) if val_loss_col else None

    return epochs, val_iou, val_loss


# ─────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────

def plot_overlay(
    dataframes:    List[Optional[pd.DataFrame]],
    variant_names: List[str],
    out_path:      Path,
) -> None:
    """
    Generate 2-panel overlay figure: val IoU (left) + val Loss (right).

    Args:
        dataframes:    list of 4 DataFrames (or None for missing variants)
        variant_names: list of 4 variant names e.g. ["A","B","C","D"]
        out_path:      save path for PNG
    """
    # IEEE double-column: 7.16 in wide × 2.8 in tall
    fig, (ax_iou, ax_loss) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    best_val_iou: Dict[str, float] = {}

    for df, name, colour, marker, label in zip(
        dataframes, variant_names, _COLOURS, _MARKERS, _LABELS
    ):
        if df is None:
            continue

        epochs, val_iou, val_loss = _extract_series(df)
        if epochs is None:
            continue

        label_short = label  # full label in legend

        if val_iou is not None:
            # Mark best epoch with a star
            best_ep_idx = int(np.argmax(val_iou))
            best_val_iou[name] = float(val_iou[best_ep_idx])

            ax_iou.plot(
                epochs, val_iou,
                linestyle="-", marker=marker,
                color=colour, linewidth=1.5,
                markersize=3, markevery=max(1, len(epochs) // 8),
                label=f"{label_short}",
                zorder=3,
            )
            ax_iou.plot(
                epochs[best_ep_idx], val_iou[best_ep_idx],
                marker="*", color=colour,
                markersize=10, zorder=5, linestyle="none",
            )

        if val_loss is not None:
            ax_loss.plot(
                epochs, val_loss,
                linestyle="-", marker=marker,
                color=colour, linewidth=1.5,
                markersize=3, markevery=max(1, len(epochs) // 8),
                label=f"{label_short}",
                zorder=3,
            )

    # ── IoU panel ────────────────────────────────────────────
    ax_iou.set_xlabel("Epoch")
    ax_iou.set_ylabel("Validation IoU")
    ax_iou.set_title("Validation IoU  ↑")
    ax_iou.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax_iou.legend(fontsize=7.5, loc="lower right", ncol=1)

    # Annotate best IoU for each variant
    for name, colour in zip(variant_names, _COLOURS):
        if name in best_val_iou:
            ax_iou.text(
                0.98, 0.04 + variant_names.index(name) * 0.065,
                f"{name}: {best_val_iou[name]:.4f}",
                transform=ax_iou.transAxes,
                fontsize=7, ha="right", va="bottom",
                color=colour,
            )

    # ── Loss panel ───────────────────────────────────────────
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation Loss")
    ax_loss.set_title("Validation Loss  ↓")
    ax_loss.legend(fontsize=7.5, loc="upper right", ncol=1)

    fig.suptitle("Training Curves — All Ablation Variants (Validation)")
    plt.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot overlaid validation curves for all 4 ablation variants"
    )
    # Convenience: auto-discover CSVs from a shared directory
    ap.add_argument(
        "--curves_dir", type=str, default=None,
        help="Directory containing variant_A_scalars.csv … variant_D_scalars.csv "
             "(produced by export_tb_curves.py). If given, overrides --csv_* args.",
    )
    # Explicit per-variant paths
    ap.add_argument("--csv_A", type=str, default=None)
    ap.add_argument("--csv_B", type=str, default=None)
    ap.add_argument("--csv_C", type=str, default=None)
    ap.add_argument("--csv_D", type=str, default=None)
    # Output
    ap.add_argument(
        "--out_dir", type=str, default="results/paper_figures",
        help="Output directory for the combined figure",
    )
    ap.add_argument(
        "--out_name", type=str, default="fig_training_curves_overlay.png",
        help="Output filename",
    )
    args = ap.parse_args()

    _set_pub_style()

    variant_names = ["A", "B", "C", "D", "E"]

    # Resolve CSV paths
    if args.curves_dir:
        curves_dir = Path(args.curves_dir)
        csv_paths = {
            v: curves_dir / f"variant_{v}_scalars.csv"
            for v in variant_names
        }
    else:
        csv_paths = {
            "A": Path(args.csv_A) if args.csv_A else None,
            "B": Path(args.csv_B) if args.csv_B else None,
            "C": Path(args.csv_C) if args.csv_C else None,
            "D": Path(args.csv_D) if args.csv_D else None,
            "E": None,  # no standalone --csv_E arg; use --curves_dir instead
        }

    dataframes = []
    for v in variant_names:
        p = csv_paths.get(v)
        df = _load_csv(p) if p else None
        dataframes.append(df)

    loaded = sum(df is not None for df in dataframes)
    if loaded == 0:
        print("ERROR: No CSV files found. Run export_tb_curves.py first.")
        sys.exit(1)
    print(f"Loaded {loaded}/{len(variant_names)} variant CSVs.")

    out_path = Path(args.out_dir) / args.out_name
    plot_overlay(dataframes, variant_names, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
