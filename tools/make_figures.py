#!/usr/bin/env python3
"""
Generate publication-ready figures + small tables for EO flood segmentation.

Outputs:
  results/paper_figures/
    fig1_sar_example.png
    fig2_label_example.png
    fig3_sar_vs_label.png
    fig4_dataset_grid.png
    fig5_class_distribution.png
    fig6_training_curves.png
    fig7_predictions_from_tb.png   (if TB images exist)

  results/paper_tables/
    class_distribution.csv
    run_summary.md
    run_summary.csv

Usage:
  python tools/make_paper_figures.py \
    --data_root data/sen1floods11 \
    --runs_dir checkpoints/variant_B/runs \
    --metrics_csv results/curves/variant_B_metrics.csv \
    --variant B \
    --out_dir results
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError as e:
    raise SystemExit("Missing dependency: rasterio. Install it in your env.") from e

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Missing dependency: pillow (PIL). Install it in your env.") from e

# TensorBoard event reader (ships with tensorboard)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


# -----------------------------
# Helpers
# -----------------------------

def set_pub_style():
    """
    Apply publication-quality matplotlib style.

    Targets IEEE TGRS / Remote Sensing of Environment / IGARSS standards:
      - Serif fonts (Times New Roman / DejaVu Serif) matching journal body text
      - Single column: 3.5 in  |  double column: 7.16 in
      - Minimum 300 DPI for raster exports
      - Font sizes >= 8 pt for all visible text
      - Clean spines (top/right removed), light grid
      - mathtext uses STIX (matches LaTeX Computer Modern closely)

    Call this once at the top of any script before creating figures.
    """
    plt.rcParams.update({
        # ── Resolution ─────────────────────────────────────────────────────
        "figure.dpi":       160,   # screen preview
        "savefig.dpi":      300,   # final export (IEEE minimum)
        "savefig.format":   "png",
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.05,

        # ── Fonts ───────────────────────────────────────────────────────────
        # Prefer Times New Roman; fall back to DejaVu Serif then sans-serif.
        # Use rcParams["font.family"] = "sans-serif" for conferences like IGARSS
        # that accept Helvetica/Arial, but "serif" for full journal submissions.
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "Palatino",
                              "Georgia", "serif"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "figure.titlesize":  12,
        "figure.titleweight": "bold",

        # ── Math text ───────────────────────────────────────────────────────
        # 'stix' matches LaTeX Computer Modern closely for inline math labels
        "mathtext.fontset":  "stix",

        # ── Lines and markers ───────────────────────────────────────────────
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        "patch.linewidth":   0.8,

        # ── Axes ────────────────────────────────────────────────────────────
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,

        # ── Grid ────────────────────────────────────────────────────────────
        "axes.grid":         True,
        "grid.alpha":        0.20,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "grid.color":        "#888888",

        # ── Legend ──────────────────────────────────────────────────────────
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.7",
        "legend.borderpad":   0.4,

        # ── Figure ──────────────────────────────────────────────────────────
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",

        # ── Ticks ───────────────────────────────────────────────────────────
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "xtick.minor.size":  2.0,
        "ytick.minor.size":  2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def percentile_stretch(x: np.ndarray, lo=2.0, hi=98.0) -> np.ndarray:
    x = x.astype(np.float32)
    a, b = np.nanpercentile(x, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0)


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()  # (C, H, W)
    return arr


def extract_event_name(stem: str) -> str:
    # chip id format seen: "Bolivia_103757_S1Hand" etc.
    # event name is prefix before first "_"
    return stem.split("_")[0]


def find_handlabeled_paths(data_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Returns (s1_paths, label_paths) for HandLabeled dataset.
    """
    base = data_root / "flood_events" / "HandLabeled"
    s1_dir = base / "S1Hand"
    lab_dir = base / "LabelHand"

    if not s1_dir.exists() or not lab_dir.exists():
        raise FileNotFoundError(
            f"Expected HandLabeled folders:\n"
            f"  {s1_dir}\n  {lab_dir}\n"
            f"Check --data_root (you passed: {data_root})."
        )

    s1_paths = sorted(s1_dir.glob("*_S1Hand.tif"))
    label_paths = sorted(lab_dir.glob("*_LabelHand.tif"))

    if len(s1_paths) == 0 or len(label_paths) == 0:
        raise RuntimeError(
            f"No files found. Counts:\n"
            f"  S1Hand: {len(s1_paths)}\n"
            f"  LabelHand: {len(label_paths)}"
        )
    return s1_paths, label_paths


def pair_s1_and_label(s1_paths: List[Path], label_paths: List[Path]) -> List[Tuple[Path, Path]]:
    lab_map = {p.stem.replace("_LabelHand", ""): p for p in label_paths}
    pairs = []
    for s1 in s1_paths:
        chip = s1.stem.replace("_S1Hand", "")
        lab = lab_map.get(chip)
        if lab is not None:
            pairs.append((s1, lab))
    if not pairs:
        raise RuntimeError("Could not pair any S1Hand with LabelHand files (chip ids mismatch).")
    return pairs


# -----------------------------
# Figure generators
# -----------------------------

def fig1_sar_example(pair: Tuple[Path, Path], out: Path, title_prefix: str):
    s1_path, _ = pair
    s1 = read_tif(s1_path)  # (2,H,W) typically

    # Use VV = band 1 by convention here
    vv = s1[0]
    vv_s = percentile_stretch(vv)

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = plt.gca()
    ax.imshow(vv_s, cmap="gray")
    ax.set_title(f"{title_prefix}: Sentinel-1 SAR (VV) example")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.text(0.01, -0.08, f"Chip: {s1_path.name}", transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "fig1_sar_example.png", bbox_inches="tight")
    plt.close(fig)


def fig2_label_example(pair: Tuple[Path, Path], out: Path, title_prefix: str):
    _, lab_path = pair
    lab = read_tif(lab_path)[0]  # (H,W)

    # Labels expected: 0,1,2 (ignore may exist)
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = plt.gca()
    im = ax.imshow(lab, vmin=0, vmax=2)
    ax.set_title(f"{title_prefix}: Hand-labeled flood mask example")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class")
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["0: non-flood", "1: flood", "2: permanent water"])

    ax.text(0.01, -0.08, f"Chip: {lab_path.name}", transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "fig2_label_example.png", bbox_inches="tight")
    plt.close(fig)


def fig3_sar_vs_label(pair: Tuple[Path, Path], out: Path, title_prefix: str):
    s1_path, lab_path = pair
    s1 = read_tif(s1_path)
    lab = read_tif(lab_path)[0]

    vv = percentile_stretch(s1[0])

    fig = plt.figure(figsize=(10.5, 4.6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(vv, cmap="gray")
    ax1.set_title("Sentinel-1 SAR (VV)")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")

    ax2 = plt.subplot(1, 2, 2)
    im = ax2.imshow(lab, vmin=0, vmax=2)
    ax2.set_title("Hand-labeled flood extent")
    ax2.set_xlabel("x (pixels)")
    ax2.set_ylabel("y (pixels)")

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Class")
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["0", "1", "2"])

    fig.suptitle(f"{title_prefix}: Example training pair", y=1.02, fontsize=15)
    plt.tight_layout()
    fig.savefig(out / "fig3_sar_vs_label.png", bbox_inches="tight")
    plt.close(fig)


def fig_overlay_sar_label(pair: Tuple[Path, Path], out: Path, title_prefix: str):
    """
    Optional but strongly recommended for papers:
    SAR grayscale + semi-transparent flood overlay.
    """
    s1_path, lab_path = pair
    s1 = read_tif(s1_path)
    lab = read_tif(lab_path)[0]

    vv = percentile_stretch(s1[0])

    # Overlay: flood=1 (and optionally permanent water=2)
    flood = (lab == 1).astype(np.float32)
    permw = (lab == 2).astype(np.float32)

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = plt.gca()
    ax.imshow(vv, cmap="gray")
    ax.imshow(flood, alpha=0.35)   # default colormap is fine
    ax.imshow(permw, alpha=0.20)   # a second overlay
    ax.set_title(f"{title_prefix}: SAR with flood overlay")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.text(0.01, -0.08, f"Chip: {s1_path.name}", transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "figX_sar_label_overlay.png", bbox_inches="tight")
    plt.close(fig)


def fig4_dataset_grid(pairs: List[Tuple[Path, Path]], out: Path, title_prefix: str, n: int = 9, seed: int = 42):
    """
    Grid of SAR samples across different events if possible.
    """
    random.seed(seed)

    # Try to pick diverse event names
    by_event: Dict[str, List[Tuple[Path, Path]]] = {}
    for s1, lab in pairs:
        event = extract_event_name(s1.stem)
        by_event.setdefault(event, []).append((s1, lab))

    # Sample up to n events; if fewer events, sample multiple from same
    events = list(by_event.keys())
    random.shuffle(events)

    chosen: List[Tuple[Path, Path]] = []
    for ev in events:
        chosen.append(random.choice(by_event[ev]))
        if len(chosen) >= n:
            break
    while len(chosen) < n:
        chosen.append(random.choice(pairs))

    cols = 3
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(11.0, 7.8))
    for i, (s1_path, _) in enumerate(chosen[:n], start=1):
        s1 = read_tif(s1_path)
        vv = percentile_stretch(s1[0])

        ax = plt.subplot(rows, cols, i)
        ax.imshow(vv, cmap="gray")
        ev = extract_event_name(s1_path.stem)
        ax.set_title(ev, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{title_prefix}: Dataset diversity (SAR VV samples)", y=0.98, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "fig4_dataset_grid.png", bbox_inches="tight")
    plt.close(fig)


def fig5_class_distribution(pairs: List[Tuple[Path, Path]], out: Path, tables_out: Path, max_chips: int = 120):
    """
    Compute pixel counts of label classes on a subset for a quick imbalance summary.
    """
    sample = pairs[:max_chips] if len(pairs) > max_chips else pairs

    counts = {0: 0, 1: 0, 2: 0, "other": 0}
    total = 0

    for _, lab_path in sample:
        lab = read_tif(lab_path)[0]
        # ignore negatives if any
        lab_valid = lab[lab >= 0]
        total += lab_valid.size
        for k in (0, 1, 2):
            counts[k] += int((lab_valid == k).sum())
        counts["other"] += int(((lab_valid != 0) & (lab_valid != 1) & (lab_valid != 2)).sum())

    # Save CSV table
    ensure_dir(tables_out)
    csv_path = tables_out / "class_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "pixel_count", "fraction"])
        for k in [0, 1, 2, "other"]:
            frac = (counts[k] / total) if total > 0 else 0.0
            w.writerow([k, counts[k], f"{frac:.6f}"])

    # Plot bar chart
    labels = ["0: non-flood", "1: flood", "2: perm. water", "other/ignore"]
    values = [counts[0], counts[1], counts[2], counts["other"]]

    fig = plt.figure(figsize=(8.2, 5.2))
    ax = plt.gca()
    ax.bar(labels, values)
    ax.set_title("Class distribution (pixel counts) — HandLabeled subset")
    ax.set_ylabel("Pixels")
    ax.set_xlabel("Label class")
    ax.tick_params(axis="x", rotation=15)

    # annotate fractions
    for i, v in enumerate(values):
        frac = v / total if total else 0.0
        ax.text(i, v * 1.01 + 1, f"{frac*100:.2f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(out / "fig5_class_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig6_training_curves(metrics_csv: Optional[Path], out: Path, title_prefix: str):
    """
    Paper curve figure from metrics csv (epoch, train_loss, val_loss, train_iou, val_iou).
    If your file has different columns, it will try to infer.
    """
    if metrics_csv is None or not metrics_csv.exists():
        print("[WARN] metrics_csv not found. Skipping fig6_training_curves.")
        return

    import pandas as pd
    df = pd.read_csv(metrics_csv)

    # Try common column patterns
    colmap_candidates = [
        # (epoch, train_loss, val_loss, train_iou, val_iou)
        ("epoch", "train_loss", "val_loss", "train_iou", "val_iou"),
        ("Epoch", "Train/loss", "Val/loss", "Train/iou", "Val/iou"),
        ("epoch", "loss_train", "loss_val", "iou_train", "iou_val"),
    ]

    chosen = None
    for cols in colmap_candidates:
        if all(c in df.columns for c in cols):
            chosen = cols
            break

    # Fallback: best effort
    if chosen is None:
        # find an epoch-ish column
        epoch_col = next((c for c in df.columns if "epoch" in c.lower()), None)
        if epoch_col is None:
            print("[WARN] Could not find an epoch column in metrics_csv. Skipping fig6.")
            return

        def find_col(keys: List[str]) -> Optional[str]:
            for k in keys:
                for c in df.columns:
                    if k in c.lower():
                        return c
            return None

        train_loss = find_col(["train_loss", "loss_train", "train/loss", "train loss"])
        val_loss   = find_col(["val_loss", "loss_val", "val/loss", "val loss"])
        train_iou  = find_col(["train_iou", "iou_train", "train/iou", "train iou"])
        val_iou    = find_col(["val_iou", "iou_val", "val/iou", "val iou"])
        chosen = (epoch_col, train_loss, val_loss, train_iou, val_iou)

    epoch_col, tr_loss, va_loss, tr_iou, va_iou = chosen

    if tr_loss is None or va_loss is None or tr_iou is None or va_iou is None:
        print("[WARN] Missing one of required columns for curves. Skipping fig6.")
        print("       Found columns:", df.columns.tolist())
        return

    epochs = df[epoch_col].values

    # One figure with two panels is common in papers; but you asked “figures”, so we’ll save ONE combined figure.
    fig = plt.figure(figsize=(10.8, 4.6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, df[tr_loss].values, label="train")
    ax1.plot(epochs, df[va_loss].values, label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, df[tr_iou].values, label="train")
    ax2.plot(epochs, df[va_iou].values, label="val")
    ax2.set_title("IoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()

    fig.suptitle(f"{title_prefix}: Training curves", y=1.02, fontsize=15)
    plt.tight_layout()
    fig.savefig(out / "fig6_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig7_predictions_from_tb(runs_dir: Path, out: Path, title_prefix: str, tag: str = "val/sar_pred_label", max_k: int = 6):
    """
    Extract images from TensorBoard event logs (like val/sar_pred_label) and save a montage.
    If TB is missing or tag doesn't exist, prints warning.
    """
    if EventAccumulator is None:
        print("[WARN] tensorboard EventAccumulator not available. Skipping fig7.")
        return
    if runs_dir is None or not runs_dir.exists():
        print("[WARN] runs_dir not found. Skipping fig7.")
        return

    ea = EventAccumulator(str(runs_dir))
    ea.Reload()
    tags = ea.Tags().get("images", [])
    if tag not in tags:
        print(f"[WARN] TB image tag '{tag}' not found. Available image tags: {tags}. Skipping fig7.")
        return

    imgs = ea.Images(tag)
    if not imgs:
        print(f"[WARN] No images for tag '{tag}'. Skipping fig7.")
        return

    # Take last max_k
    chosen = imgs[-max_k:]

    pil_imgs = []
    for im in chosen:
        pil = Image.open(BytesIO(im.encoded_image_string)).convert("RGB")
        pil_imgs.append(pil)

    # Make a montage (2 x ceil(k/2))
    k = len(pil_imgs)
    cols = 2
    rows = math.ceil(k / cols)

    fig = plt.figure(figsize=(10.8, 5.2))
    for i, pil in enumerate(pil_imgs, start=1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(np.asarray(pil))
        ax.set_title(f"Step {chosen[i-1].step}", fontsize=11)
        ax.axis("off")

    fig.suptitle(f"{title_prefix}: Qualitative predictions (TensorBoard tag: {tag})", y=0.98, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "fig7_predictions_from_tb.png", bbox_inches="tight")
    plt.close(fig)


def save_run_summary(metrics_csv: Optional[Path], out_tables: Path, variant: str):
    """
    Create a small paper-friendly summary table from metrics file (best val IoU, last epoch, etc.).
    """
    ensure_dir(out_tables)

    rows = []
    if metrics_csv and metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)

        epoch_col = next((c for c in df.columns if "epoch" in c.lower()), df.columns[0])

        def find_col(keys: List[str]) -> Optional[str]:
            for k in keys:
                for c in df.columns:
                    if k in c.lower():
                        return c
            return None

        val_iou = find_col(["val_iou", "val/iou", "iou_val"])
        val_loss = find_col(["val_loss", "val/loss", "loss_val"])
        train_iou = find_col(["train_iou", "train/iou", "iou_train"])
        train_loss = find_col(["train_loss", "train/loss", "loss_train"])

        if val_iou is not None:
            best_idx = int(df[val_iou].astype(float).idxmax())
            best_epoch = int(df.loc[best_idx, epoch_col])
            best_val_iou = float(df.loc[best_idx, val_iou])
        else:
            best_epoch = None
            best_val_iou = None

        last_epoch = int(df.loc[df.index[-1], epoch_col])

        rows.append({
            "variant": variant,
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_val_iou": best_val_iou,
            "val_loss_last": float(df.loc[df.index[-1], val_loss]) if val_loss else None,
            "train_loss_last": float(df.loc[df.index[-1], train_loss]) if train_loss else None,
            "train_iou_last": float(df.loc[df.index[-1], train_iou]) if train_iou else None,
        })
    else:
        rows.append({"variant": variant})

    # Save CSV
    csv_path = out_tables / "run_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Save MD
    md_path = out_tables / "run_summary.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(sorted(rows[0].keys())) + " |\n")
        f.write("|" + "|".join(["---"] * len(sorted(rows[0].keys()))) + "|\n")
        for r in rows:
            f.write("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in sorted(rows[0].keys())) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g., data/sen1floods11")
    ap.add_argument("--runs_dir", type=str, default=None, help="e.g., checkpoints/variant_B/runs")
    ap.add_argument("--metrics_csv", type=str, default=None, help="e.g., results/curves/variant_B_metrics.csv")
    ap.add_argument("--variant", type=str, default="B", help="Variant label (A/B/C)")
    ap.add_argument("--out_dir", type=str, default="results", help="Base output dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid_n", type=int, default=9)
    ap.add_argument("--class_max_chips", type=int, default=120)
    args = ap.parse_args()

    set_pub_style()

    data_root = Path(args.data_root)
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else None

    out_base = Path(args.out_dir)
    figs_out = ensure_dir(out_base / "paper_figures")
    tables_out = ensure_dir(out_base / "paper_tables")

    title_prefix = f"TerrainFlood-UQ | Variant {args.variant}"

    s1_paths, lab_paths = find_handlabeled_paths(data_root)
    pairs = pair_s1_and_label(s1_paths, lab_paths)

    random.seed(args.seed)
    example_pair = random.choice(pairs)

    print(f"[INFO] Found {len(pairs)} paired chips.")
    print(f"[INFO] Example chip: {example_pair[0].name}")

    # 1–3: one example
    fig1_sar_example(example_pair, figs_out, title_prefix)
    fig2_label_example(example_pair, figs_out, title_prefix)
    fig3_sar_vs_label(example_pair, figs_out, title_prefix)

    # Bonus overlay (highly recommended)
    fig_overlay_sar_label(example_pair, figs_out, title_prefix)

    # 4: dataset grid
    fig4_dataset_grid(pairs, figs_out, title_prefix, n=args.grid_n, seed=args.seed)

    # 5: class distribution + table
    fig5_class_distribution(pairs, figs_out, tables_out, max_chips=args.class_max_chips)

    # 6: training curves
    fig6_training_curves(metrics_csv, figs_out, title_prefix)

    # 7: predictions from TensorBoard (if available)
    if runs_dir is not None:
        fig7_predictions_from_tb(runs_dir, figs_out, title_prefix, tag="val/sar_pred_label", max_k=6)

    # Small paper table
    save_run_summary(metrics_csv, tables_out, args.variant)

    print("[DONE] Saved figures to:", figs_out)
    print("[DONE] Saved tables  to:", tables_out)


if __name__ == "__main__":
    main()