#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# TensorBoard event reader (ships with tensorboard)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ─────────────────────────────────────────────────────────────
# Column name normalisation
# Maps raw TensorBoard tag names → CSV column names that are
# expected by tools/make_figures.py:fig6_training_curves().
# The first colmap_candidates tuple there is:
#   ("epoch", "train_loss", "val_loss", "train_iou", "val_iou")
# so we must match those exact strings.
# ─────────────────────────────────────────────────────────────
_TAG_TO_COL: Dict[str, str] = {
    "Loss/train": "train_loss",
    "Loss/val":   "val_loss",
    "IoU/train":  "train_iou",
    "IoU/val":    "val_iou",
    "LR":         "lr",
}


def _load_scalars(logdir: Path, tags: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Returns:
      dict[tag] = list of (step, value) sorted by step
    """
    ea = EventAccumulator(
        str(logdir),
        size_guidance={
            "scalars": 0,  # load all
        },
    )
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    data: Dict[str, List[Tuple[int, float]]] = {}

    for tag in tags:
        if tag not in available:
            # keep empty so we can warn later
            data[tag] = []
            continue
        events = ea.Scalars(tag)
        data[tag] = sorted([(e.step, float(e.value)) for e in events], key=lambda x: x[0])

    return data


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(
    out_csv: Path,
    data: Dict[str, List[Tuple[int, float]]],
    step_col: str = "epoch",
) -> None:
    """
    Writes a wide CSV with columns: <step_col>, <tag1>, <tag2>, ...
    Aligns by step; missing values left blank.

    Args:
        out_csv:  destination path
        data:     {column_name: [(step, value), ...]}
        step_col: name for the step column (default "epoch" to match
                  make_figures.py:fig6_training_curves expectations)
    """
    # collect union of steps
    steps = sorted({s for series in data.values() for (s, _) in series})
    by_tag = {tag: dict(series) for tag, series in data.items()}

    _ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([step_col, *data.keys()])
        for s in steps:
            row = [s]
            for tag in data.keys():
                v = by_tag[tag].get(s, "")
                row.append(v)
            w.writerow(row)


def _plot_series(
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Dict[str, List[Tuple[int, float]]],
) -> None:
    _ensure_dir(out_png.parent)

    plt.figure(figsize=(7, 4.2), dpi=160)
    for name, pts in series.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if any(len(v) > 0 for v in series.values()):
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, required=True, help="TensorBoard run dir (contains events.out.tfevents...)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder (will create curves/ and figures/)")
    ap.add_argument("--variant", type=str, default="B")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    out_dir = Path(args.out_dir)

    # Tags logged by train.py via add_scalars / add_scalar.
    # add_scalars("Loss", {"train":..., "val":...}) → "Loss/train", "Loss/val"
    # add_scalars("IoU",  {"train":..., "val":...}) → "IoU/train",  "IoU/val"
    # add_scalar("LR", ...) → "LR"
    tags = [
        "Loss/train",
        "Loss/val",
        "IoU/train",
        "IoU/val",
        "LR",
    ]

    data = _load_scalars(logdir, tags)

    # Normalise column names so make_figures.py:fig6_training_curves()
    # can parse the CSV without manual intervention.
    # Raw TB tags  →  normalised CSV columns:
    #   Loss/train  →  train_loss
    #   Loss/val    →  val_loss
    #   IoU/train   →  train_iou
    #   IoU/val     →  val_iou
    #   LR          →  lr
    data_normed = {_TAG_TO_COL.get(k, k): v for k, v in data.items()}

    # Save CSV  (step column written as "epoch" to match make_figures.py)
    out_csv = out_dir / "curves" / f"variant_{args.variant}_scalars.csv"
    _write_csv(out_csv, data_normed, step_col="epoch")

    # Plots  (use data_normed so series keys are clean readable labels)
    _plot_series(
        out_dir / "figures" / f"variant_{args.variant}_loss.png",
        title=f"Training curves — Variant {args.variant} (Loss)",
        xlabel="Epoch",
        ylabel="Loss",
        series={
            "train": data_normed.get("train_loss", []),
            "val":   data_normed.get("val_loss",   []),
        },
    )

    _plot_series(
        out_dir / "figures" / f"variant_{args.variant}_iou.png",
        title=f"Training curves — Variant {args.variant} (IoU)",
        xlabel="Epoch",
        ylabel="IoU",
        series={
            "train": data_normed.get("train_iou", []),
            "val":   data_normed.get("val_iou",   []),
        },
    )

    # Also save LR if present
    if data_normed.get("lr"):
        _plot_series(
            out_dir / "figures" / f"variant_{args.variant}_lr.png",
            title=f"Learning rate — Variant {args.variant}",
            xlabel="Epoch",
            ylabel="LR",
            series={"lr": data_normed["lr"]},
        )

    # Quick console summary (report missing using original tag names)
    missing = [t for t, v in data.items() if len(v) == 0]
    print(f"Saved: {out_csv}")
    print(f"Saved: {(out_dir / 'figures').resolve()}")
    if missing:
        print("WARNING: Missing tags (may be normal if your logger uses different names):")
        for t in missing:
            print(f"  - {t}")
        print("\nIf tags are different, run: python -c \"from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA; ea=EA(r'LOGDIR'); ea.Reload(); print(ea.Tags())\"")


if __name__ == "__main__":
    main()