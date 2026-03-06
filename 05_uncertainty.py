"""
Phase 4 — Uncertainty Estimation
==================================
File: pipeline/05_uncertainty.py

MC Dropout inference:
  - T=20 stochastic forward passes with dropout active
  - Predictive mean  → flood probability map
  - Predictive variance → uncertainty map
  - Trust mask → pixels where model is confident

Calibration:
  - Expected Calibration Error (ECE)
  - Reliability diagram
  - Brier score

Usage:
  python pipeline/05_uncertainty.py \
      --checkpoint checkpoints/variant_D_TIMESTAMP/best.pt \
      --data_root  data/sen1floods11 \
      --output_dir results/uncertainty \
      --T 20
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm

import sys
import importlib.util

_unc_root = Path(__file__).parent


def _import_module(alias: str, file_path: str):
    """Load a Python file with a numeric prefix as a named module."""
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "model"   not in sys.modules:
    _import_module("model",   str(_unc_root / "03_model.py"))
if "dataset" not in sys.modules:
    _import_module("dataset", str(_unc_root / "02_dataset.py"))

from model   import build_model     # noqa: E402
from dataset import get_dataloaders  # noqa: E402


# ─────────────────────────────────────────────────────────────
# 1.  MC Dropout inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def mc_dropout_inference(
    model:    torch.nn.Module,
    loader:   torch.utils.data.DataLoader,
    device:   torch.device,
    T:        int = 20,
) -> list[dict]:
    """
    Runs T stochastic forward passes per batch with dropout active.

    Returns a dict of per-chip results:
      mean_prob  : (H, W) float32 — predictive mean flood probability
      variance   : (H, W) float32 — predictive variance (uncertainty)
      label      : (H, W) int16   — ground truth label
      event      : str            — flood event name
      chip_id    : str            — chip identifier
    """
    # Keep dropout active, disable batchnorm updates
    model.eval()
    model.enable_dropout()

    results = []

    for batch in tqdm(loader, desc=f"MC Dropout (T={T})"):
        images   = batch["image"].to(device)   # (B, 6, H, W)
        labels   = batch["label"]              # (B, H, W) — keep on CPU
        events   = batch["event"]
        chip_ids = batch["chip_id"]

        B = images.shape[0]
        mc_preds = []

        # T stochastic forward passes
        for t in range(T):
            logits = model(images)                        # (B, 1, H, W)
            probs  = torch.sigmoid(logits).squeeze(1)     # (B, H, W)
            mc_preds.append(probs.cpu())

        mc_preds = torch.stack(mc_preds, dim=0)           # (T, B, H, W)

        mean_prob = mc_preds.mean(dim=0)                  # (B, H, W)
        variance  = mc_preds.var(dim=0)                   # (B, H, W)

        for i in range(B):
            results.append({
                "mean_prob": mean_prob[i].numpy(),        # (H, W)
                "variance":  variance[i].numpy(),         # (H, W)
                "label":     labels[i].numpy(),           # (H, W)
                "event":     events[i],
                "chip_id":   chip_ids[i],
            })

    return results


# ─────────────────────────────────────────────────────────────
# 1b. Logit collection (single forward pass — for temperature scaling)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits(
    model:  torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single deterministic forward pass; returns (logits, labels) for all
    valid pixels concatenated. Used as calibration set for temperature scaling.

    Returns:
        logits: (N,) float64 — raw model logits (before sigmoid)
        labels: (N,) float64 — binary ground truth (0/1, -1 excluded)
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Collecting logits"):
        images = batch["image"].to(device)   # (B, 6, H, W)
        labels = batch["label"]              # (B, H, W)

        logits = model(images).squeeze(1).cpu().numpy()  # (B, H, W)

        for i in range(images.shape[0]):
            valid = labels[i].numpy() != -1
            all_logits.append(logits[i][valid].flatten())
            all_labels.append(labels[i].numpy()[valid].astype(np.float64))

    return np.concatenate(all_logits).astype(np.float64), \
           np.concatenate(all_labels).astype(np.float64)


# ─────────────────────────────────────────────────────────────
# 2.  Trust mask
# ─────────────────────────────────────────────────────────────

def compute_trust_mask(
    variance:          np.ndarray,
    uncertainty_threshold: float = 0.05,
) -> np.ndarray:
    """
    Binary trust mask: 1 = confident prediction, 0 = uncertain.

    threshold=0.05 means: if predictive variance > 0.05, don't trust.
    Variance of 0.05 corresponds to ±0.22 std dev in flood probability —
    i.e. the model disagrees significantly across MC passes.

    This mask is used in 06_exposure.py to gate population counting:
    only pixels inside the trust mask contribute to exposure estimates.

    Args:
        variance: (H, W) predictive variance from MC Dropout
        uncertainty_threshold: max allowed variance for a trusted pixel

    Returns:
        trust_mask: (H, W) bool array
    """
    return variance <= uncertainty_threshold


# ─────────────────────────────────────────────────────────────
# 3.  Calibration metrics
# ─────────────────────────────────────────────────────────────

def compute_ece(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ignore_value: int = -1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error (ECE).

    A well-calibrated model: if it says 70% flood probability,
    70% of those pixels should actually be flooded.
    ECE measures the average deviation from perfect calibration.

    Lower ECE = better calibrated. Target: ECE < 0.05.

    Args:
        probs:  (N,) predicted probabilities flattened
        labels: (N,) binary ground truth labels flattened
        n_bins: number of calibration bins

    Returns:
        ece:         scalar ECE value
        bin_accs:    (n_bins,) mean accuracy per bin
        bin_confs:   (n_bins,) mean confidence per bin
    """
    # Remove ignored pixels
    valid   = labels != ignore_value
    probs   = probs[valid].astype(np.float32)
    labels  = labels[valid].astype(np.float32)

    bins     = np.linspace(0, 1, n_bins + 1)
    bin_accs  = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (probs >= bins[i]) & (probs < bins[i + 1])
        if in_bin.sum() > 0:
            bin_accs[i]  = labels[in_bin].mean()
            bin_confs[i] = probs[in_bin].mean()
            bin_sizes[i] = in_bin.sum()

    # Weighted average over bins
    total    = bin_sizes.sum()
    ece      = (bin_sizes / (total + 1e-8) * np.abs(bin_accs - bin_confs)).sum()

    return float(ece), bin_accs, bin_confs


def compute_brier_score(
    probs:  np.ndarray,
    labels: np.ndarray,
    ignore_value: int = -1,
) -> float:
    """
    Brier score = mean squared error between probabilities and labels.
    Lower is better. Perfect = 0.0, random = 0.25.
    """
    valid  = labels != ignore_value
    probs  = probs[valid].astype(np.float32)
    labels = labels[valid].astype(np.float32)
    return float(np.mean((probs - labels) ** 2))


# ─────────────────────────────────────────────────────────────
# 3b. Temperature scaling
# ─────────────────────────────────────────────────────────────

def temperature_scale(
    logits:       np.ndarray,
    labels:       np.ndarray,
    ignore_value: int = -1,
) -> float:
    """
    Finds the optimal temperature T that minimises NLL on a calibration set.

    Post-hoc calibration: the model weights are frozen; only T is optimised.
    Usage after finding T:
        calibrated_probs = apply_temperature_scaling(logits, T)

    Args:
        logits:       (N,) raw model logits (before sigmoid)
        labels:       (N,) binary ground truth (0 or 1)
        ignore_value: pixels to exclude (default -1)

    Returns:
        T: optimal temperature scalar in [0.1, 10.0]
           T > 1 → softer (more uncertain) probabilities → better calibrated
           T < 1 → sharper (more confident) probabilities
    """
    valid  = labels != ignore_value
    logits = logits[valid].astype(np.float64)
    labels = labels[valid].astype(np.float64)

    def nll(T: float) -> float:
        """Negative log-likelihood of binary cross-entropy at temperature T."""
        scaled = logits / T
        probs  = 1.0 / (1.0 + np.exp(-scaled))
        probs  = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(
            labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)
        ))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def apply_temperature_scaling(
    logits: np.ndarray,
    T:      float,
) -> np.ndarray:
    """
    Applies temperature scaling to raw logits. Returns calibrated probabilities.

    Args:
        logits: (N,) or (H, W) raw model logits
        T:      temperature from temperature_scale()

    Returns:
        probs: same shape as logits, float32 in [0, 1]
    """
    scaled = logits.astype(np.float64) / T
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 4.  Plots
# ─────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    bin_accs:  np.ndarray,
    bin_confs: np.ndarray,
    ece:       float,
    out_path:  str,
    title:     str = "Reliability Diagram",
):
    """
    Reliability diagram (calibration curve).
    Perfect calibration = diagonal line.
    Area between curve and diagonal = miscalibration.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

    # Model calibration
    n_bins = len(bin_accs)
    bin_centers = np.linspace(1/(2*n_bins), 1 - 1/(2*n_bins), n_bins)
    ax.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.6,
           color="steelblue", label="Model", edgecolor="white")
    ax.plot(bin_centers, bin_accs, "o-", color="steelblue", linewidth=2)

    ax.set_xlabel("Confidence (predicted probability)", fontsize=12)
    ax.set_ylabel("Accuracy (fraction truly flooded)", fontsize=12)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_uncertainty_map(
    mean_prob:  np.ndarray,
    variance:   np.ndarray,
    trust_mask: np.ndarray,
    label:      np.ndarray,
    chip_id:    str,
    out_path:   str,
):
    """
    4-panel figure per chip:
      [SAR post VV | flood probability | uncertainty | trust mask vs label]
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    panels = [
        (mean_prob,                    "Flood probability",  "RdYlBu_r", 0, 1),
        (variance,                     "Predictive variance","hot_r",     0, variance.max()),
        (trust_mask.astype(float),     "Trust mask",         "Greens",    0, 1),
        ((label > 0).astype(float),    "Ground truth",       "Blues",     0, 1),
    ]

    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Chip: {chip_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 5.  Per-event aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_results(results: list) -> dict:
    """
    Aggregates per-chip results into per-event and overall metrics.

    Returns nested dict:
      {
        "overall": {ece, brier, mean_variance, coverage},
        "per_event": {
          "Cambodia": {ece, brier, mean_variance, n_chips},
          ...
        }
      }
    """
    from collections import defaultdict
    event_data = defaultdict(lambda: {"probs": [], "labels": [], "variances": []})

    for r in results:
        valid = r["label"] != -1
        event_data[r["event"]]["probs"].append(r["mean_prob"][valid].flatten())
        event_data[r["event"]]["labels"].append(r["label"][valid].flatten())
        event_data[r["event"]]["variances"].append(r["variance"][valid].flatten())

    per_event = {}
    all_probs, all_labels, all_vars = [], [], []

    for event, data in event_data.items():
        probs  = np.concatenate(data["probs"])
        labels = np.concatenate(data["labels"])
        variances = np.concatenate(data["variances"])

        ece, _, _ = compute_ece(probs, labels)
        brier     = compute_brier_score(probs, labels)

        per_event[event] = {
            "ece":           round(ece, 4),
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

    overall_ece, bin_accs, bin_confs = compute_ece(all_probs, all_labels)
    overall_brier = compute_brier_score(all_probs, all_labels)

    return {
        "overall": {
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


# ─────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────

def run_uncertainty(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MC samples T={args.T}\n")

    out_dir = Path(args.output_dir)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt    = torch.load(args.checkpoint, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"Loaded Variant {variant} from epoch {ckpt['epoch']} "
          f"(best val IoU={ckpt['best_iou']:.4f})\n")

    # Data — run on test split (Bolivia OOD)
    _, val_loader, test_loader = get_dataloaders(
        data_root   = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Temperature scaling (optional) ─────────────────────────
    temperature = 1.0
    if args.calibrate:
        print("Finding optimal temperature on val set (Ecuador + Paraguay)...")
        val_logits, val_labels = collect_logits(model, val_loader, device)
        temperature = temperature_scale(val_logits, val_labels)
        print(f"  Optimal temperature T = {temperature:.4f}")
        temp_path = out_dir / "temperature.json"
        temp_path.write_text(json.dumps({"temperature": temperature}, indent=2))
        print(f"  Saved → {temp_path}")

    # MC Dropout inference
    results = mc_dropout_inference(model, test_loader, device, T=args.T)
    print(f"\nProcessed {len(results)} chips\n")

    # Apply temperature scaling to mean_prob if calibration was run
    if args.calibrate and temperature != 1.0:
        print(f"  Applying temperature scaling (T={temperature:.4f}) to mean_prob...")
        for r in results:
            # Recover approximate logits from probabilities via logit transform
            p = np.clip(r["mean_prob"], 1e-6, 1.0 - 1e-6)
            approx_logits = np.log(p / (1.0 - p))
            r["mean_prob_raw"]     = r["mean_prob"].copy()   # save original
            r["mean_prob"]         = apply_temperature_scaling(approx_logits, temperature)

    # Add trust masks
    for r in results:
        r["trust_mask"] = compute_trust_mask(
            r["variance"], uncertainty_threshold=args.uncertainty_threshold
        )

    # Aggregate metrics
    print("Computing calibration metrics...")
    metrics = aggregate_results(results)

    print(f"\n{'='*50}")
    if args.calibrate:
        # Compute uncalibrated ECE for comparison
        all_probs_raw = []
        all_labels_raw = []
        for r in results:
            valid = r["label"] != -1
            raw_p = r.get("mean_prob_raw", r["mean_prob"])
            all_probs_raw.append(raw_p[valid].flatten())
            all_labels_raw.append(r["label"][valid].flatten())
        ece_before, _, _ = compute_ece(
            np.concatenate(all_probs_raw), np.concatenate(all_labels_raw)
        )
        print(f"ECE before calibration : {ece_before:.4f}")
        print(f"ECE after  calibration : {metrics['overall']['ece']:.4f}  (T={temperature:.4f})")
    else:
        print(f"Overall ECE   : {metrics['overall']['ece']:.4f}  (target < 0.05)")
    print(f"Overall Brier : {metrics['overall']['brier']:.4f}")
    print(f"Mean variance : {metrics['overall']['mean_variance']:.4f}")
    print(f"{'='*50}")
    print("\nPer-event ECE:")
    for event, m in metrics["per_event"].items():
        print(f"  {event:<15} ECE={m['ece']:.4f}  Brier={m['brier']:.4f}")

    # Save metrics JSON
    metrics_path = out_dir / "uncertainty_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved → {metrics_path}")

    # Reliability diagram
    bin_accs  = np.array(metrics["calibration"]["bin_accs"])
    bin_confs = np.array(metrics["calibration"]["bin_confs"])
    plot_reliability_diagram(
        bin_accs, bin_confs, metrics["overall"]["ece"],
        out_path=str(out_dir / "reliability_diagram.png"),
        title=f"Variant {variant} — Test Set (Bolivia OOD)",
    )

    # Per-chip uncertainty maps (first N chips)
    print(f"\nSaving uncertainty maps for first {args.n_maps} chips...")
    for r in results[:args.n_maps]:
        fname = f"{r['event']}_{r['chip_id']}_uncertainty.png"
        plot_uncertainty_map(
            mean_prob  = r["mean_prob"],
            variance   = r["variance"],
            trust_mask = r["trust_mask"],
            label      = r["label"],
            chip_id    = r["chip_id"],
            out_path   = str(out_dir / "maps" / fname),
        )

    print(f"\nDone. Results in {out_dir}")
    return results, metrics


# ─────────────────────────────────────────────────────────────
# 7.  Argument parser
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MC Dropout Uncertainty Estimation")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to best.pt checkpoint")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/uncertainty")
    p.add_argument("--T",            type=int, default=20,
                   help="Number of MC Dropout forward passes")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--uncertainty_threshold", type=float, default=0.05,
                   help="Max variance for trusted pixel")
    p.add_argument("--n_maps",       type=int, default=10,
                   help="Number of uncertainty map figures to save")
    p.add_argument("--calibrate",    action="store_true",
                   help="Find optimal temperature T on val set and apply to test predictions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_uncertainty(args)