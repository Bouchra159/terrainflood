#!/usr/bin/env python3
"""
Classical SAR Flood Detection Baseline — Otsu Thresholding
File: tools/otsu_baseline.py

Provides two classical baselines for direct comparison against neural models:

  1. VV-post Otsu         — Otsu threshold on VV_post backscatter intensity.
                            Flooded areas have lower SAR return (open water
                            acts as a specular reflector away from sensor).
                            Pixels below threshold → flood.

  2. Change-detection Otsu — Otsu threshold on ΔVV = (VV_pre − VV_post).
                             Large positive change = backscatter dropped after
                             event = surface changed to water. Standard
                             operational approach used by Copernicus EMS.

Both are applied per-chip (chip-level Otsu) AND globally (threshold computed
over all test chips combined — mimics a "pre-calibrated" operational system).

Output: results/eval_otsu/metrics.json  (same schema as eval.py outputs so
        ablation_metrics.json can be extended with an Otsu row)

Usage:
  python tools/otsu_baseline.py \\
      --data_root data/sen1floods11 \\
      --output_dir results/eval_otsu \\
      --split test

  # Also evaluate on val set:
  python tools/otsu_baseline.py \\
      --data_root data/sen1floods11 \\
      --output_dir results/eval_otsu_val \\
      --split val
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


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


if "dataset_otsu" not in sys.modules:
    _import_module("dataset_otsu", str(_root / "02_dataset.py"))

from dataset_otsu import get_dataloaders  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Band indices (must match 02_dataset.py band table)
# ─────────────────────────────────────────────────────────────
CH_VV_PRE  = 0
CH_VH_PRE  = 1
CH_VV_POST = 2
CH_VH_POST = 3


# ─────────────────────────────────────────────────────────────
# Otsu threshold implementation (no sklearn dependency)
# ─────────────────────────────────────────────────────────────

def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Compute Otsu's threshold for a 1-D array of values.

    Maximises between-class variance (equivalent to minimising
    within-class variance). Standard Sentinel-1 flood detection approach.

    Args:
        values:  1-D float array of valid pixel values
        n_bins:  number of histogram bins (default 256)

    Returns:
        Scalar threshold value (float)
    """
    counts, bin_edges = np.histogram(values, bins=n_bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return float(np.median(values))

    # Otsu criterion: maximise between-class variance
    weight1 = np.cumsum(counts) / total
    weight2 = 1.0 - weight1
    mean_cum = np.cumsum(counts * bin_centres)
    mean_total = mean_cum[-1] / total
    mean1 = np.where(weight1 > 0, mean_cum / (weight1 * total + 1e-12), 0.0)
    mean2 = np.where(weight2 > 0,
                     (mean_total - weight1 * mean1) / (weight2 + 1e-12),
                     0.0)
    between_class_var = weight1 * weight2 * (mean1 - mean2) ** 2
    return float(bin_centres[np.argmax(between_class_var)])


# ─────────────────────────────────────────────────────────────
# Segmentation metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    pred:        np.ndarray,   # (H, W) bool — predicted flood
    label:       np.ndarray,   # (H, W) int  — ground truth
    ignore_val:  int = -1,
) -> dict:
    """
    Compute IoU, F1, precision, recall, accuracy over valid pixels only.

    Flood = label ∈ {1, 2}  (permanent water treated as flood, same as eval.py).
    """
    valid  = label != ignore_val
    target = ((label == 1) | (label == 2))[valid]
    pred_v = pred[valid]

    tp = int(( pred_v &  target).sum())
    fp = int(( pred_v & ~target).sum())
    fn = int((~pred_v &  target).sum())
    tn = int((~pred_v & ~target).sum())

    iou       = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-6)

    return {
        "iou":       round(float(iou),       4),
        "f1":        round(float(f1),        4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "accuracy":  round(float(accuracy),  4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────────
# Main baseline runner
# ─────────────────────────────────────────────────────────────

def run_otsu_baseline(
    data_root:   str,
    output_dir:  str,
    split:       str = "test",
    num_workers: int = 2,
) -> None:
    """
    Run Otsu and change-detection baselines on the requested split.

    Saves results/eval_otsu/metrics.json with the same outer schema as
    eval.py so it can be compared directly in the ablation table.

    Args:
        data_root:   Sen1Floods11 root directory
        output_dir:  where to write metrics.json
        split:       "test" (Bolivia OOD) or "val" (Paraguay)
        num_workers: dataloader workers
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root   = data_root,
        batch_size  = 1,   # process chip by chip for per-chip Otsu
        num_workers = num_workers,
        patch_size  = None,
        pin_memory  = False,
    )
    loader = test_loader if split == "test" else val_loader

    print(f"\nOtsu Baseline  |  split={split}")
    print(f"Output: {out_dir}\n")

    # ── Collect pixel values across all chips (for global Otsu) ──
    all_vv_post:  list[float] = []
    all_vv_delta: list[float] = []

    chips: list[dict] = []

    for batch in loader:
        images  = batch["image"].numpy()    # (1, 6, H, W) normalised
        labels  = batch["label"].numpy()    # (1, H, W)
        chip_id = batch["chip_id"][0]
        event   = batch["event"][0]

        img = images[0]     # (6, H, W)
        lbl = labels[0]     # (H, W)

        vv_post   = img[CH_VV_POST]               # (H, W)
        vv_delta  = img[CH_VV_PRE] - img[CH_VV_POST]  # (H, W)  pre − post

        valid_mask = (lbl != -1)
        if valid_mask.sum() == 0:
            continue

        # Per-chip Otsu: VV_post (flood = below threshold → dark = water)
        vv_vals   = vv_post[valid_mask].astype(np.float64)
        thresh_vv = otsu_threshold(vv_vals)
        pred_vv   = (vv_post < thresh_vv)   # dark SAR return → flood

        # Per-chip Otsu: change detection (flood = large positive ΔVV)
        dv_vals    = vv_delta[valid_mask].astype(np.float64)
        thresh_dv  = otsu_threshold(dv_vals)
        pred_delta = (vv_delta > thresh_dv)   # backscatter dropped → flood

        m_vv    = compute_metrics(pred_vv,    lbl)
        m_delta = compute_metrics(pred_delta, lbl)

        chips.append({
            "chip_id":         chip_id,
            "event":           event,
            "thresh_vv_post":  round(thresh_vv,  4),
            "thresh_vv_delta": round(thresh_dv,  4),
            "vv_post_otsu":    m_vv,
            "change_detect":   m_delta,
        })

        # Accumulate for global Otsu
        all_vv_post.extend(vv_vals.tolist())
        all_vv_delta.extend(dv_vals.tolist())

        print(f"  {chip_id:<32}  "
              f"VV-Otsu IoU={m_vv['iou']:.3f}  "
              f"ΔVV-Otsu IoU={m_delta['iou']:.3f}  "
              f"thresh_vv={thresh_vv:.3f}")

    if not chips:
        print("ERROR: no chips found. Check --data_root and --split.")
        return

    # ── Global Otsu (all test chips pooled) ──────────────────
    global_thresh_vv    = otsu_threshold(np.array(all_vv_post,  dtype=np.float64))
    global_thresh_delta = otsu_threshold(np.array(all_vv_delta, dtype=np.float64))

    print(f"\nGlobal thresholds:  VV_post={global_thresh_vv:.3f}  "
          f"ΔVV={global_thresh_delta:.3f}")

    # Re-evaluate with global threshold
    agg_global_vv    = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    agg_global_delta = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    agg_perchip_vv   = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    agg_perchip_dv   = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    loader2 = test_loader if split == "test" else val_loader

    for batch in loader2:
        images  = batch["image"].numpy()
        labels  = batch["label"].numpy()
        img     = images[0]
        lbl     = labels[0]

        vv_post   = img[CH_VV_POST]
        vv_delta  = img[CH_VV_PRE] - img[CH_VV_POST]
        valid_mask = (lbl != -1)
        if valid_mask.sum() == 0:
            continue

        # Global threshold predictions
        pred_g_vv    = (vv_post  < global_thresh_vv)
        pred_g_delta = (vv_delta > global_thresh_delta)

        mg_vv    = compute_metrics(pred_g_vv,    lbl)
        mg_delta = compute_metrics(pred_g_delta, lbl)

        for k in ("tp", "fp", "fn", "tn"):
            agg_global_vv[k]    += mg_vv[k]
            agg_global_delta[k] += mg_delta[k]

        # Per-chip aggregation (from chips list)
        chip_id = batch["chip_id"][0]
        c = next((x for x in chips if x["chip_id"] == chip_id), None)
        if c:
            for k in ("tp", "fp", "fn", "tn"):
                agg_perchip_vv[k] += c["vv_post_otsu"][k]
                agg_perchip_dv[k] += c["change_detect"][k]

    def _from_agg(agg: dict) -> dict:
        tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
        iou       = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-6)
        return {
            "iou":       round(iou,       4),
            "f1":        round(f1,        4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "accuracy":  round(accuracy,  4),
            **agg,
            "ece": None, "brier": None, "mean_variance": None,
        }

    overall_vv_perchip    = _from_agg(agg_perchip_vv)
    overall_delta_perchip = _from_agg(agg_perchip_dv)
    overall_vv_global     = _from_agg(agg_global_vv)
    overall_delta_global  = _from_agg(agg_global_delta)

    results = {
        "split":        split,
        "n_chips":      len(chips),
        "methods": {
            "vv_post_otsu_perchip": {
                "description": "Otsu threshold on normalised VV_post, per-chip",
                "overall": overall_vv_perchip,
            },
            "change_detect_perchip": {
                "description": "Otsu threshold on (VV_pre - VV_post), per-chip",
                "overall": overall_delta_perchip,
            },
            "vv_post_otsu_global": {
                "description": "Otsu threshold on VV_post, pooled over all chips",
                "global_threshold": round(global_thresh_vv, 4),
                "overall": overall_vv_global,
            },
            "change_detect_global": {
                "description": "Otsu threshold on (VV_pre - VV_post), pooled over all chips",
                "global_threshold": round(global_thresh_delta, 4),
                "overall": overall_delta_global,
            },
        },
        "per_chip": chips,
    }

    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'Method':<35} {'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
    print("-" * 63)
    for name, method in results["methods"].items():
        m = method["overall"]
        print(f"  {name:<33} {m['iou']:>7.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f}")

    print(
        "\nReference (neural models — from ablation_metrics.json):"
        "\n  Variant A (SAR-only CNN)          IoU=0.4082"
        "\n  Variant D (HAND gate + MC-Drop)   IoU=0.6898"
    )
    print(
        "\nInterpretation:"
        "\n  Otsu IoU < Variant A IoU  → deep learning adds clear value over thresholding."
        "\n  VV-post vs ΔVV comparison  → shows which SAR signal is more discriminative."
        "\n  Per-chip vs global Otsu    → quantifies benefit of per-image calibration."
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical Otsu / change-detection SAR flood baseline"
    )
    p.add_argument("--data_root",   type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",  type=str, default="results/eval_otsu")
    p.add_argument("--split",       type=str, default="test",
                   choices=["test", "val"],
                   help="test = Bolivia OOD (15 chips), val = Paraguay (67 chips)")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_otsu_baseline(
        data_root   = args.data_root,
        output_dir  = args.output_dir,
        split       = args.split,
        num_workers = args.num_workers,
    )
