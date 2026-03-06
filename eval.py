"""
Phase 6 — Evaluation
======================
File: eval.py

Full evaluation pipeline for TerrainFlood-UQ.

  1. Segmentation metrics: IoU, F1, Precision, Recall (per event + overall)
  2. Uncertainty metrics:  ECE, Brier score, mean variance
  3. Ablation table:       compare all 4 variants side-by-side
  4. Figures:
       - Reliability diagram
       - Coverage-accuracy curve
       - Per-event IoU bar chart
       - Ablation comparison bars
       - Per-chip flood map figures (first N chips)

Usage:
  # Evaluate variant D on test set (Bolivia OOD)
  python eval.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/eval_D \\
      --T 20 \\
      --n_maps 10

  # Full ablation: evaluate all 4 variants
  python eval.py \\
      --ablation \\
      --checkpoints_dir checkpoints \\
      --data_root data/sen1floods11 \\
      --output_dir results/ablation
"""

import json
import csv
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from plots import (
    plot_flood_map,
    plot_reliability_diagram,
    plot_coverage_accuracy,
    plot_iou_bar_chart,
    plot_ablation_table,
    plot_risk_coverage_curve,
)
from trust_mask import compute_trust_mask


# ─────────────────────────────────────────────────────────────
# Numeric-prefix module imports
# ─────────────────────────────────────────────────────────────

def _import_module(alias: str, file_path: str):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_root = Path(__file__).parent
if "model"   not in sys.modules:
    _import_module("model",   str(_root / "03_model.py"))
if "dataset" not in sys.modules:
    _import_module("dataset", str(_root / "02_dataset.py"))

from model   import build_model      # noqa: E402
from dataset import get_dataloaders  # noqa: E402

_unc = _import_module("uncertainty_mod", str(_root / "05_uncertainty.py"))
mc_dropout_inference = _unc.mc_dropout_inference
compute_ece          = _unc.compute_ece
compute_brier_score  = _unc.compute_brier_score


# ─────────────────────────────────────────────────────────────
# 1.  Segmentation metrics
# ─────────────────────────────────────────────────────────────

def compute_segmentation_metrics(
    probs:        np.ndarray,
    labels:       np.ndarray,
    threshold:    float = 0.5,
    ignore_value: int   = -1,
) -> dict:
    """
    Binary segmentation metrics for the flood class.

    Args:
        probs:    (N,) float32 predicted flood probabilities
        labels:   (N,) int ground truth (0 or 1, -1 = ignore)
        threshold: decision threshold (default 0.5)
        ignore_value: pixels to skip

    Returns:
        dict: iou, f1, precision, recall, accuracy
    """
    valid  = labels != ignore_value
    probs  = probs[valid].astype(np.float32)
    labels = labels[valid].astype(np.int32)

    preds = (probs > threshold).astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    iou       = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "iou":       round(iou,       4),
        "f1":        round(f1,        4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "accuracy":  round(accuracy,  4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────────
# 1b. Risk-Coverage curve
# ─────────────────────────────────────────────────────────────

def compute_risk_coverage_curve(
    probs:        np.ndarray,
    variance:     np.ndarray,
    labels:       np.ndarray,
    n_thresholds: int = 50,
    ignore_value: int = -1,
) -> dict:
    """
    At each uncertainty threshold τ (swept from 0 → max_variance):
      coverage = fraction of valid pixels with variance ≤ τ  (trusted)
      risk     = 1 − IoU computed only on the trusted pixels

    A model with good uncertainty estimates achieves low risk even at high
    coverage — confident predictions should be correct.

    AURC (area under risk-coverage) summarises the curve; lower = better.

    Args:
        probs:        (N,) predicted flood probabilities
        variance:     (N,) predictive variance (from MC Dropout)
        labels:       (N,) int ground truth (0/1, ignore_value excluded)
        n_thresholds: number of τ steps
        ignore_value: label value to skip

    Returns:
        dict with keys:
          thresholds : (n_thresholds,) float — variance thresholds used
          coverage   : (n_thresholds,) float — fraction of trusted pixels
          risk       : (n_thresholds,) float — 1 − IoU on trusted pixels
          aurc       : float — area under risk-coverage curve (lower is better)
    """
    valid    = labels != ignore_value
    probs    = probs[valid].astype(np.float32)
    variance = variance[valid].astype(np.float32)
    labels   = labels[valid].astype(np.float32)

    max_var    = float(variance.max()) or 1.0
    thresholds = np.linspace(0.0, max_var, n_thresholds + 1)[1:]  # skip τ=0

    coverages = []
    risks     = []

    for tau in thresholds:
        trusted = variance <= tau
        n_trusted = trusted.sum()
        if n_trusted == 0:
            coverages.append(0.0)
            risks.append(1.0)
            continue

        cov   = float(n_trusted) / len(probs)
        preds = (probs[trusted] > 0.5).astype(np.float32)
        labs  = labels[trusted]

        inter = float(((preds == 1) & (labs == 1)).sum())
        union = float(((preds == 1) | (labs == 1)).sum())
        iou   = inter / max(union, 1.0)

        coverages.append(cov)
        risks.append(1.0 - iou)

    coverages = np.array(coverages)
    risks     = np.array(risks)

    # AURC via trapezoidal integration
    # Convention: sort by ascending coverage so trapz integrates correctly
    order     = np.argsort(coverages)
    aurc      = float(np.trapz(risks[order], coverages[order]))

    return {
        "thresholds": thresholds.tolist(),
        "coverage":   coverages.tolist(),
        "risk":       risks.tolist(),
        "aurc":       round(aurc, 4),
    }


# ─────────────────────────────────────────────────────────────
# 2.  Evaluate a single checkpoint
# ─────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path:  str,
    data_root:  str,
    device:     torch.device,
    T:          int    = 20,
    split:      str    = "test",
    batch_size: int    = 4,
    num_workers: int   = 4,
    uncertainty_threshold: float = 0.05,
) -> tuple[list, dict]:
    """
    Runs MC Dropout inference on the requested split and computes all metrics.

    Args:
        ckpt_path:  path to best.pt checkpoint
        data_root:  root of Sen1Floods11 dataset
        device:     torch device
        T:          MC Dropout forward passes
        split:      "test" (Bolivia OOD) or "val"

    Returns:
        (results, metrics)
        results: list of per-chip dicts from mc_dropout_inference()
        metrics: nested dict with overall / per_event / calibration
    """
    ckpt    = torch.load(ckpt_path, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    print(f"  Loaded Variant {variant}  epoch={ckpt['epoch']}  "
          f"best_iou={ckpt.get('best_iou', 0.0):.4f}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root   = data_root,
        batch_size  = batch_size,
        num_workers = num_workers,
    )
    loader = test_loader if split == "test" else val_loader

    results = mc_dropout_inference(model, loader, device, T=T)

    # Per-event metrics
    from collections import defaultdict
    event_data = defaultdict(lambda: {"probs": [], "labels": [], "variances": []})

    for r in results:
        valid = r["label"] != -1
        r["trust_mask"] = compute_trust_mask(r["variance"], uncertainty_threshold)
        event_data[r["event"]]["probs"].append(r["mean_prob"][valid].flatten())
        event_data[r["event"]]["labels"].append(r["label"][valid].flatten())
        event_data[r["event"]]["variances"].append(r["variance"][valid].flatten())

    per_event   = {}
    all_probs   = []
    all_labels  = []
    all_vars    = []

    for event, data in event_data.items():
        probs     = np.concatenate(data["probs"])
        labels    = np.concatenate(data["labels"])
        variances = np.concatenate(data["variances"])

        seg   = compute_segmentation_metrics(probs, labels)
        ece_v, _, _ = compute_ece(probs, labels)
        brier = compute_brier_score(probs, labels)

        per_event[event] = {
            **seg,
            "ece":           round(ece_v, 4),
            "brier":         round(brier, 4),
            "mean_variance": round(float(variances.mean()), 4),
            "n_chips":       len(data["probs"]),
        }
        all_probs.append(probs)
        all_labels.append(labels)
        all_vars.append(variances)

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_vars   = np.concatenate(all_vars)

    overall_seg  = compute_segmentation_metrics(all_probs, all_labels)
    overall_ece, bin_accs, bin_confs = compute_ece(all_probs, all_labels)
    overall_brier = compute_brier_score(all_probs, all_labels)

    metrics = {
        "variant": variant,
        "overall": {
            **overall_seg,
            "ece":           round(overall_ece, 4),
            "brier":         round(overall_brier, 4),
            "mean_variance": round(float(all_vars.mean()), 4),
        },
        "per_event": per_event,
        "calibration": {
            "bin_accs":  bin_accs.tolist(),
            "bin_confs": bin_confs.tolist(),
        },
    }

    return results, metrics


# ─────────────────────────────────────────────────────────────
# 3.  Ablation table
# ─────────────────────────────────────────────────────────────

def build_ablation_table(
    checkpoints_dir: str,
    data_root:       str,
    device:          torch.device,
    T:               int = 20,
) -> tuple[dict, list[dict]]:
    """
    Evaluates all 4 variants (A/B/C/D) and builds a comparison table.

    Expects checkpoint files at:
        checkpoints_dir/variant_A/best.pt
        checkpoints_dir/variant_B/best.pt
        checkpoints_dir/variant_C/best.pt
        checkpoints_dir/variant_D/best.pt

    Returns:
        (ablation_dict, rows)
        ablation_dict: {variant: overall_metrics}
        rows: list of dicts for CSV export
    """
    ablation   = {}
    csv_rows   = []
    ckpt_dir   = Path(checkpoints_dir)

    for variant in ["A", "B", "C", "D"]:
        ckpt_path = ckpt_dir / f"variant_{variant}" / "best.pt"
        if not ckpt_path.exists():
            print(f"  Variant {variant}: checkpoint not found at {ckpt_path}, skipping.")
            continue

        print(f"\n--- Evaluating Variant {variant} ---")
        _, metrics = evaluate_checkpoint(
            str(ckpt_path), data_root, device, T=T
        )
        ablation[variant] = metrics["overall"]

        row = {"variant": variant}
        row.update(metrics["overall"])
        csv_rows.append(row)

    return ablation, csv_rows


# ─────────────────────────────────────────────────────────────
# 4.  Single-checkpoint evaluation run
# ─────────────────────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    """Full evaluation pipeline for a single checkpoint."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    figs_dir = out_dir / "figures"
    maps_dir = out_dir / "maps"
    figs_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation config for reproducibility
    import json as _json
    (out_dir / "eval_config.json").write_text(
        _json.dumps(vars(args), indent=2)
    )

    print(f"\nDevice      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Output dir  : {out_dir}\n")

    results, metrics = evaluate_checkpoint(
        ckpt_path             = args.checkpoint,
        data_root             = args.data_root,
        device                = device,
        T                     = args.T,
        batch_size            = args.batch_size,
        num_workers           = args.num_workers,
        uncertainty_threshold = args.uncertainty_threshold,
    )

    variant = metrics["variant"]
    overall = metrics["overall"]

    print(f"\n{'='*55}")
    print(f"  Variant {variant} — Test Set (Bolivia OOD)")
    print(f"{'='*55}")
    print(f"  IoU       : {overall['iou']:.4f}")
    print(f"  F1        : {overall['f1']:.4f}")
    print(f"  Precision : {overall['precision']:.4f}")
    print(f"  Recall    : {overall['recall']:.4f}")
    print(f"  ECE       : {overall['ece']:.4f}  (target < 0.05)")
    print(f"  Brier     : {overall['brier']:.4f}")
    print(f"  Mean var  : {overall['mean_variance']:.4f}")
    print(f"{'='*55}")
    print("\nPer-event IoU:")
    for event, m in metrics["per_event"].items():
        print(f"  {event:<15} IoU={m['iou']:.4f}  F1={m['f1']:.4f}")

    # Save metrics JSON
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics → {metrics_path}")

    # Reliability diagram
    bin_accs  = np.array(metrics["calibration"]["bin_accs"])
    bin_confs = np.array(metrics["calibration"]["bin_confs"])
    plot_reliability_diagram(
        bin_accs, bin_confs, overall["ece"],
        out_path=str(figs_dir / "reliability_diagram.png"),
        title=f"Variant {variant} — Bolivia OOD",
    )

    # Per-event IoU bar chart
    plot_iou_bar_chart(
        metrics["per_event"],
        out_path=str(figs_dir / "iou_per_event.png"),
        title=f"Variant {variant} — IoU per Event",
    )

    # Coverage-accuracy curve
    _variant_descriptions = {
        "A": "Variant A (SAR only)",
        "B": "Variant B (HAND as band)",
        "C": "Variant C (HAND gate, no UQ)",
        "D": "Variant D (HAND gate + MC Dropout)",
    }
    plot_coverage_accuracy(
        results,
        out_path=str(figs_dir / "coverage_accuracy.png"),
        variant_label=_variant_descriptions.get(variant, f"Variant {variant}"),
    )

    # Risk-Coverage curve
    all_probs_rc = np.concatenate([
        r["mean_prob"][r["label"] != -1].flatten() for r in results
    ])
    all_vars_rc = np.concatenate([
        r["variance"][r["label"] != -1].flatten() for r in results
    ])
    all_labels_rc = np.concatenate([
        r["label"][r["label"] != -1].flatten() for r in results
    ])
    rc = compute_risk_coverage_curve(all_probs_rc, all_vars_rc, all_labels_rc)
    plot_risk_coverage_curve(
        rc,
        out_path=str(figs_dir / "risk_coverage.png"),
        variant_labels=[f"Variant {variant}"],
    )
    # Save RC data for potential multi-variant overlay in ablation
    import json as _json
    (out_dir / "risk_coverage.json").write_text(_json.dumps(rc, indent=2))
    print(f"  AURC: {rc['aurc']:.4f}  (lower is better)")

    # Per-chip flood maps (first N)
    print(f"\nSaving flood maps for first {args.n_maps} chips...")
    for r in results[:args.n_maps]:
        fname = f"{r['event']}_{r['chip_id']}.png"
        plot_flood_map(
            mean_prob  = r["mean_prob"],
            variance   = r["variance"],
            trust_mask = r["trust_mask"],
            label      = r["label"],
            chip_id    = r["chip_id"],
            out_path   = str(maps_dir / fname),
        )

    print(f"\nDone. Results in {out_dir}")


# ─────────────────────────────────────────────────────────────
# 5.  Ablation evaluation run
# ─────────────────────────────────────────────────────────────

def run_ablation(args: argparse.Namespace) -> None:
    """Evaluate all 4 variants and produce comparison figures."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAblation evaluation — all 4 variants")
    print(f"Checkpoints dir : {args.checkpoints_dir}")
    print(f"Output dir      : {out_dir}\n")

    ablation, csv_rows = build_ablation_table(
        checkpoints_dir = args.checkpoints_dir,
        data_root       = args.data_root,
        device          = device,
        T               = args.T,
    )

    if not ablation:
        print("No checkpoints found. Train variants A–D first.")
        return

    # Save ablation JSON
    (out_dir / "ablation_metrics.json").write_text(
        json.dumps(ablation, indent=2)
    )

    # Save ablation CSV
    csv_path = out_dir / "ablation_table.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nAblation table → {csv_path}")

    # Ablation figure
    plot_ablation_table(
        ablation,
        out_path=str(out_dir / "ablation_comparison.png"),
    )

    # Print table
    print(f"\n{'Variant':<10} {'IoU':>8} {'F1':>8} {'ECE':>8} {'Brier':>8}")
    print("-" * 46)
    for v in ["A", "B", "C", "D"]:
        if v in ablation:
            m = ablation[v]
            print(f"  {v:<8} {m['iou']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['ece']:>8.4f} {m['brier']:>8.4f}")

    print(f"\nDone. Results in {out_dir}")


# ─────────────────────────────────────────────────────────────
# 6.  Argument parser
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TerrainFlood-UQ Evaluation")

    # Mode
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation (all 4 variants) instead of single checkpoint")

    # Single-checkpoint mode
    p.add_argument("--checkpoint",      type=str, default=None,
                   help="Path to best.pt (single-variant mode)")

    # Ablation mode
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                   help="Dir containing variant_A/, variant_B/, etc.")

    # Common
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/eval")
    p.add_argument("--T",            type=int, default=20,
                   help="MC Dropout forward passes")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--uncertainty_threshold", type=float, default=0.05)
    p.add_argument("--n_maps",       type=int, default=10,
                   help="Number of flood map figures to save")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ablation:
        run_ablation(args)
    else:
        if args.checkpoint is None:
            raise ValueError("Provide --checkpoint or use --ablation flag")
        run_evaluation(args)
