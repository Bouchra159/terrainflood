#!/usr/bin/env python3
"""
Deep Ensemble Inference for TerrainFlood-UQ
File: tools/deep_ensemble.py

Implements the Lakshminarayanan et al. (2017) Deep Ensemble approach
as the gold-standard alternative to MC Dropout for uncertainty estimation.

Why Deep Ensembles work better than MC Dropout here:
  - MC Dropout variance ≈ 0.0004 (near-zero) because BatchNorm absorbs
    the dropout stochasticity during training.
  - Each ensemble member starts from a different random initialisation
    and processes training batches in a different order → genuine
    parameter space diversity → meaningful prediction disagreement
    on uncertain pixels.
  - Expected ensemble variance range: 0.01–0.10 (vs MC 0.0004)
  - Uncertainty-error correlation should be positive (r > 0.5)

Reference:
  Lakshminarayanan et al. (2017) "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles", NeurIPS 2017.

STEP 1 — Train 3 independent Variant D models with different seeds:
  (On DKUCC — run all three sbatch jobs, they are independent)

  python train.py --variant D --seed 42  \\
      --data_root data/sen1floods11 \\
      --output_dir checkpoints/variant_D_s42

  python train.py --variant D --seed 123 \\
      --data_root data/sen1floods11 \\
      --output_dir checkpoints/variant_D_s123

  python train.py --variant D --seed 456 \\
      --data_root data/sen1floods11 \\
      --output_dir checkpoints/variant_D_s456

STEP 2 — Run ensemble inference:
  python tools/deep_ensemble.py \\
      --checkpoints checkpoints/variant_D_s42/best.pt \\
                    checkpoints/variant_D_s123/best.pt \\
                    checkpoints/variant_D_s456/best.pt \\
      --data_root   data/sen1floods11 \\
      --output_dir  results/ensemble_D

STEP 3 — Validate UQ quality:
  python tools/uncertainty_error_correlation.py \\
      --ensemble_mode \\
      --checkpoints checkpoints/variant_D_s42/best.pt \\
                    checkpoints/variant_D_s123/best.pt \\
                    checkpoints/variant_D_s456/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/uncertainty_ensemble
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
from scipy.optimize import minimize_scalar


# ─────────────────────────────────────────────────────────────
# Numeric-prefix module imports
# ─────────────────────────────────────────────────────────────

_root = Path(__file__).parent.parent


def _import_module(alias: str, file_path: str):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "model_ens"   not in sys.modules:
    _import_module("model_ens",   str(_root / "03_model.py"))
if "dataset_ens" not in sys.modules:
    _import_module("dataset_ens", str(_root / "02_dataset.py"))
if "unc_ens"     not in sys.modules:
    _import_module("unc_ens",     str(_root / "05_uncertainty.py"))

from model_ens   import build_model          # noqa: E402
from dataset_ens import get_dataloaders      # noqa: E402
from unc_ens     import (                    # noqa: E402
    compute_ece,
    compute_brier_score,
    temperature_scale,
    apply_temperature_scaling,
)


# ─────────────────────────────────────────────────────────────
# Deep Ensemble class
# ─────────────────────────────────────────────────────────────

class FloodDeepEnsemble:
    """
    Deep Ensemble of independently trained FloodSegmentationModel instances.

    Each member was trained from a different random seed, providing
    genuine parameter-space diversity. Their disagreement on difficult
    pixels reflects real epistemic uncertainty.

    Args:
        checkpoint_paths: list of paths to best.pt files (3–5 models)
        device:           torch device

    Attributes:
        models:  list of loaded FloodSegmentationModel instances
        variant: model variant (must be the same for all members)
    """

    def __init__(
        self,
        checkpoint_paths: list[str],
        device:           torch.device,
    ) -> None:
        self.device  = device
        self.models  = []
        self.variant = None
        self.member_info: list[dict] = []

        for i, ckpt_path in enumerate(checkpoint_paths):
            ckpt    = torch.load(ckpt_path, map_location=device)
            config  = ckpt.get("config", {})
            variant = config.get("variant", "D")
            epoch   = ckpt.get("epoch", -1)
            best_iou = ckpt.get("best_iou", 0.0)

            if self.variant is None:
                self.variant = variant
            elif self.variant != variant:
                print(f"  [WARN] Member {i} is Variant {variant} "
                      f"but expected {self.variant} — proceeding anyway")

            model = build_model(variant=variant, pretrained=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval().to(device)
            self.models.append(model)

            info = {
                "path":     ckpt_path,
                "variant":  variant,
                "epoch":    epoch,
                "best_iou": best_iou,
                "seed":     config.get("seed", "unknown"),
            }
            self.member_info.append(info)
            print(f"  [Member {i+1}] Variant {variant}  "
                  f"epoch={epoch}  best_iou={best_iou:.4f}  "
                  f"seed={info['seed']}  ({ckpt_path})")

        print(f"\n[Ensemble] {len(self.models)} members loaded  "
              f"(Variant {self.variant})")

    @torch.no_grad()
    def predict_batch(
        self,
        x: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through all ensemble members for a single batch.

        Args:
            x: (B, 6, H, W) input tensor (already on correct device or CPU)

        Returns:
            mean_prob : (B, 1, H, W) float32 — ensemble mean flood probability
            variance  : (B, 1, H, W) float32 — ensemble variance (epistemic UQ)
            entropy   : (B, 1, H, W) float32 — predictive entropy H(p)
        """
        x = x.to(self.device)
        probs_list: list[np.ndarray] = []

        for model in self.models:
            logits = model(x)                             # (B, 1, H, W)
            probs  = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
            probs_list.append(probs)

        probs_array = np.stack(probs_list, axis=0)     # (N, B, 1, H, W)
        mean_prob   = probs_array.mean(axis=0)          # (B, 1, H, W)
        variance    = probs_array.var(axis=0)           # (B, 1, H, W)

        # Predictive entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        # Maximum at p=0.5 (uncertain), minimum at p=0 or p=1 (confident)
        p       = np.clip(mean_prob, 1e-7, 1.0 - 1e-7)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

        return (
            mean_prob.astype(np.float32),
            variance.astype(np.float32),
            entropy.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.models)


# ─────────────────────────────────────────────────────────────
# Full inference pipeline
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_ensemble_inference(
    ensemble:    FloodDeepEnsemble,
    loader:      torch.utils.data.DataLoader,
    device:      torch.device,
) -> list[dict]:
    """
    Runs all ensemble members on the full data loader.

    Returns per-chip results with the same schema as
    mc_dropout_inference() in 05_uncertainty.py, so the same
    downstream code (calibration, maps, metrics) works unchanged.

    Args:
        ensemble: FloodDeepEnsemble instance
        loader:   DataLoader (test or val split)
        device:   torch device

    Returns:
        list of dicts, one per chip:
            mean_prob : (H, W) float32 — ensemble mean flood probability
            variance  : (H, W) float32 — ensemble variance
            entropy   : (H, W) float32 — predictive entropy
            label     : (H, W) int16   — ground truth
            event     : str
            chip_id   : str
    """
    results: list[dict] = []

    total_chips = len(loader.dataset)
    print(f"Running ensemble ({len(ensemble)} members) "
          f"on {total_chips} chips...")

    for batch_idx, batch in enumerate(loader):
        images   = batch["image"]           # (B, 6, H, W) — keep on CPU
        labels   = batch["label"].numpy()   # (B, H, W)
        events   = batch["event"]
        chip_ids = batch["chip_id"]

        mean_prob, variance, entropy = ensemble.predict_batch(images)
        # Shapes: (B, 1, H, W)

        B = images.shape[0]
        for i in range(B):
            cid = chip_ids[i]
            results.append({
                "mean_prob": mean_prob[i, 0],   # (H, W)
                "variance":  variance[i, 0],    # (H, W)
                "entropy":   entropy[i, 0],     # (H, W)
                "label":     labels[i],         # (H, W)
                "event":     events[i],
                "chip_id":   cid,
            })
            print(f"  [{batch_idx * B + i + 1:3d}/{total_chips}] "
                  f"{cid:<38}  "
                  f"mean_var={variance[i,0].mean():.5f}  "
                  f"mean_H={entropy[i,0].mean():.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_ensemble_metrics(
    results: list[dict],
    temperature: float = 1.0,
) -> dict:
    """
    Computes segmentation + calibration metrics from ensemble results.

    Args:
        results:     list of per-chip dicts from run_ensemble_inference()
        temperature: temperature scaling value (1.0 = no calibration)

    Returns:
        nested dict: overall + per_event + calibration
    """
    all_probs:    list[np.ndarray] = []
    all_labels:   list[np.ndarray] = []
    all_vars:     list[np.ndarray] = []
    all_entropy:  list[np.ndarray] = []
    tp_total = fp_total = fn_total = tn_total = 0

    for r in results:
        label    = r["label"]                    # (H, W)
        prob     = r["mean_prob"]                # (H, W)
        var      = r["variance"]                 # (H, W)
        ent      = r.get("entropy", np.zeros_like(prob))

        valid    = label != -1
        probs_v  = prob[valid].astype(np.float32)
        labels_v = (label[valid] > 0).astype(np.float32)  # flood = 1 or 2 → 1
        vars_v   = var[valid].astype(np.float32)
        ents_v   = ent[valid].astype(np.float32)

        # Apply temperature scaling if requested
        if temperature != 1.0:
            logits_v = np.log(np.clip(probs_v, 1e-6, 1 - 1e-6) /
                              (1 - np.clip(probs_v, 1e-6, 1 - 1e-6)))
            probs_v  = apply_temperature_scaling(logits_v, temperature)

        preds_v = (probs_v > 0.5).astype(np.int32)
        labs_v  = labels_v.astype(np.int32)

        tp_total += int(((preds_v == 1) & (labs_v == 1)).sum())
        fp_total += int(((preds_v == 1) & (labs_v == 0)).sum())
        fn_total += int(((preds_v == 0) & (labs_v == 1)).sum())
        tn_total += int(((preds_v == 0) & (labs_v == 0)).sum())

        all_probs.append(probs_v)
        all_labels.append(labels_v)
        all_vars.append(vars_v)
        all_entropy.append(ents_v)

    all_probs   = np.concatenate(all_probs)
    all_labels  = np.concatenate(all_labels)
    all_vars    = np.concatenate(all_vars)
    all_entropy = np.concatenate(all_entropy)

    iou       = tp_total / max(tp_total + fp_total + fn_total, 1)
    precision = tp_total / max(tp_total + fp_total, 1)
    recall    = tp_total / max(tp_total + fn_total, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy  = (tp_total + tn_total) / max(
        tp_total + fp_total + fn_total + tn_total, 1)

    ece, bin_accs, bin_confs = compute_ece(all_probs, all_labels.astype(int))
    brier = compute_brier_score(all_probs, all_labels.astype(int))

    return {
        "overall": {
            "iou":           round(float(iou),       4),
            "f1":            round(float(f1),         4),
            "precision":     round(float(precision),  4),
            "recall":        round(float(recall),     4),
            "accuracy":      round(float(accuracy),   4),
            "ece":           round(float(ece),        4),
            "brier":         round(float(brier),      4),
            "mean_variance": round(float(all_vars.mean()),    5),
            "max_variance":  round(float(all_vars.max()),     5),
            "mean_entropy":  round(float(all_entropy.mean()), 4),
            "tp": tp_total, "fp": fp_total,
            "fn": fn_total, "tn": tn_total,
        },
        "calibration": {
            "bin_accs":  bin_accs.tolist(),
            "bin_confs": bin_confs.tolist(),
        },
        "temperature": temperature,
    }


# ─────────────────────────────────────────────────────────────
# Temperature calibration for ensemble
# ─────────────────────────────────────────────────────────────

def calibrate_ensemble(
    ensemble:    FloodDeepEnsemble,
    val_loader:  torch.utils.data.DataLoader,
    device:      torch.device,
) -> float:
    """
    Finds the optimal temperature T for the ensemble using the validation set.

    Args:
        ensemble:   FloodDeepEnsemble
        val_loader: validation DataLoader (Paraguay)
        device:     torch device

    Returns:
        float: optimal temperature T
    """
    print("Calibrating ensemble on validation set (Paraguay)...")
    val_results = run_ensemble_inference(ensemble, val_loader, device)

    all_probs:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for r in val_results:
        valid = r["label"] != -1
        all_probs.append(r["mean_prob"][valid].flatten())
        all_labels.append((r["label"][valid] > 0).astype(np.float64))

    probs  = np.concatenate(all_probs).astype(np.float64)
    labels = np.concatenate(all_labels).astype(np.float64)

    # Convert probabilities back to logits for temperature_scale
    logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) /
                    (1 - np.clip(probs, 1e-7, 1 - 1e-7)))

    T = temperature_scale(logits, labels)
    print(f"  Optimal temperature: T = {T:.4f}")
    return float(T)


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────

def plot_variance_comparison(
    mc_variance:  float,
    ens_variance: float,
    out_path:     Path,
) -> None:
    """
    Bar chart comparing MC Dropout vs Ensemble mean variance.
    Illustrates the UQ improvement.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    bars = ax.bar(
        ["MC Dropout\n(Variant D)", f"Deep Ensemble\n({3} members)"],
        [mc_variance, ens_variance],
        color=["#4C72B0", "#C44E52"],
        alpha=0.85, edgecolor="white", width=0.5,
    )
    for bar, val in zip(bars, [mc_variance, ens_variance]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(mc_variance, ens_variance) * 0.02,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Mean predictive variance", fontsize=11)
    ax.set_title("Uncertainty Quality: MC Dropout vs Deep Ensemble\n"
                 "(higher = more informative uncertainty)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(mc_variance, ens_variance) * 1.25)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Figure → {out_path}")


def plot_reliability_diagram_ensemble(
    bin_accs:  list[float],
    bin_confs: list[float],
    ece:       float,
    out_path:  Path,
    n_members: int = 3,
) -> None:
    """Reliability diagram for ensemble predictions."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    bins = np.array(bin_confs)
    accs = np.array(bin_accs)
    n_bins = len(bins)
    centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.bar(centers, accs, width=1 / n_bins, alpha=0.6,
           color="steelblue", label=f"Ensemble ({n_members} members)",
           edgecolor="white")
    ax.plot(centers, accs, "o-", color="steelblue", linewidth=2)

    ax.set_xlabel("Confidence",        fontsize=12)
    ax.set_ylabel("Accuracy",          fontsize=12)
    ax.set_title(f"Ensemble Reliability Diagram\nECE = {ece:.4f}", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Figure → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run_deep_ensemble(args: argparse.Namespace) -> None:
    """Full pipeline: load ensemble → calibrate → evaluate → save."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Output : {out_dir}")
    print(f"Members: {args.checkpoints}\n")

    # Build ensemble
    ensemble = FloodDeepEnsemble(
        checkpoint_paths = args.checkpoints,
        device           = device,
    )

    # Load data
    _, val_loader, test_loader = get_dataloaders(
        data_root   = args.data_root,
        batch_size  = 1,
        num_workers = args.num_workers,
        pin_memory  = False,
    )
    loader = test_loader if args.split == "test" else val_loader

    # Temperature calibration on val set
    temperature = 1.0
    if args.calibrate:
        temperature = calibrate_ensemble(ensemble, val_loader, device)
        (out_dir / "temperature.json").write_text(
            json.dumps({"temperature": temperature, "n_members": len(ensemble)},
                       indent=2)
        )

    # Inference on requested split
    print(f"\nEnsemble inference on split='{args.split}'...")
    results = run_ensemble_inference(ensemble, loader, device)

    # Metrics
    print("\nComputing metrics...")
    metrics_raw = compute_ensemble_metrics(results, temperature=1.0)
    metrics_cal = compute_ensemble_metrics(results, temperature=temperature)

    # Print summary
    raw = metrics_raw["overall"]
    cal = metrics_cal["overall"]
    print(f"\n{'='*58}")
    print(f"  Deep Ensemble ({len(ensemble)} members) — {args.split} split")
    print(f"{'='*58}")
    print(f"  IoU          : {cal['iou']:.4f}")
    print(f"  F1           : {cal['f1']:.4f}")
    print(f"  Precision    : {cal['precision']:.4f}")
    print(f"  Recall       : {cal['recall']:.4f}")
    print(f"  ECE (raw)    : {raw['ece']:.4f}")
    if temperature != 1.0:
        print(f"  ECE (cal T={temperature:.3f}): {cal['ece']:.4f}")
    print(f"  Brier        : {cal['brier']:.4f}")
    print(f"  Mean variance: {raw['mean_variance']:.5f}  "
          f"(MC Dropout baseline ≈ 0.0004)")
    print(f"  Max variance : {raw['max_variance']:.5f}")
    print(f"  Mean entropy : {raw['mean_entropy']:.4f}")
    print(f"{'='*58}")

    # Improvement vs MC Dropout
    mc_ref_variance = 0.0004
    improvement = raw["mean_variance"] / mc_ref_variance
    print(f"\n  Variance improvement vs MC Dropout: "
          f"{improvement:.0f}× "
          f"({mc_ref_variance:.4f} → {raw['mean_variance']:.5f})")
    if improvement > 10:
        print("  ✓ Ensemble provides substantially more informative uncertainty")
    elif improvement > 3:
        print("  ~ Modest improvement over MC Dropout")
    else:
        print("  ✗ Ensemble variance still low — consider more diverse seeds")

    # Save JSON
    payload = {
        "n_members":       len(ensemble),
        "checkpoints":     args.checkpoints,
        "split":           args.split,
        "temperature":     temperature,
        "metrics_uncalibrated": metrics_raw,
        "metrics_calibrated":   metrics_cal,
        "member_info":     ensemble.member_info,
    }
    out_json = out_dir / "ensemble_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON → {out_json}")

    # Figures
    plot_variance_comparison(
        mc_variance  = mc_ref_variance,
        ens_variance = raw["mean_variance"],
        out_path     = out_dir / "variance_comparison.png",
    )
    plot_reliability_diagram_ensemble(
        bin_accs  = metrics_cal["calibration"]["bin_accs"],
        bin_confs = metrics_cal["calibration"]["bin_confs"],
        ece       = cal["ece"],
        out_path  = out_dir / "reliability_diagram_ensemble.png",
        n_members = len(ensemble),
    )

    print(f"\nDone. All outputs in {out_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep Ensemble inference — gold-standard UQ for flood segmentation"
    )
    p.add_argument("--checkpoints",  type=str, nargs="+", required=True,
                   help="Paths to best.pt files (3–5 independently trained models)")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/ensemble_D")
    p.add_argument("--split",        type=str, default="test",
                   choices=["test", "val"],
                   help="test=Bolivia OOD (15 chips), val=Paraguay (67 chips)")
    p.add_argument("--calibrate",    action="store_true",
                   help="Find optimal temperature on val set before test evaluation")
    p.add_argument("--num_workers",  type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    run_deep_ensemble(_parse_args())
