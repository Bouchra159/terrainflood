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
# 0.  TTA helpers — D4 symmetry group (4 rotations × 2 flips)
# ─────────────────────────────────────────────────────────────

def tta_augment(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply the k-th element of the D4 group to a spatial tensor (B, C, H, W).

    k=0  identity
    k=1  rotate 90° CCW
    k=2  rotate 180°
    k=3  rotate 270° CCW
    k=4  horizontal flip
    k=5  flip + rotate 90°
    k=6  flip + rotate 180°
    k=7  flip + rotate 270°
    """
    flip = k >= 4
    rot  = k % 4
    if flip:
        x = torch.flip(x, dims=[-1])        # flip along W axis
    if rot > 0:
        x = torch.rot90(x, k=rot, dims=[-2, -1])
    return x


def tta_deaugment(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Exact inverse of tta_augment(·, k) for output maps (B, H, W).
    Guarantees: tta_deaugment(tta_augment(x, k), k) == x.
    """
    flip = k >= 4
    rot  = k % 4
    if rot > 0:
        x = torch.rot90(x, k=4 - rot, dims=[-2, -1])   # inverse rotation
    if flip:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def tta_inference(
    model:  torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_aug:  int = 8,
) -> list[dict]:
    """
    Test-Time Augmentation (TTA) uncertainty using the D4 symmetry group.

    Runs n_aug deterministic forward passes (no dropout), each with a
    different spatial augmentation.  Predictions are de-augmented back to
    the original orientation before being averaged.

    TTA variance captures how sensitive the model is to spatial orientation
    — a geometric proxy for epistemic uncertainty.  In practice it is
    5–50× larger than MC Dropout variance for models with bimodal logit
    distributions (temperature T ≪ 1 after calibration).

    Args:
        model:  FloodSegmentationModel — must be in eval() mode
        loader: DataLoader for the evaluation split
        device: torch device
        n_aug:  number of augmentations (8 = full D4 group, 4 = rotations only)

    Returns:
        List of per-chip dicts: {mean_prob, variance, label, event, chip_id}
    """
    model.eval()   # TTA uses NO dropout — pure geometric ensemble
    results: list[dict] = []

    for batch in tqdm(loader, desc=f"TTA ({n_aug} augmentations)"):
        images   = batch["image"].to(device)   # (B, 6, H, W)
        labels   = batch["label"]
        events   = batch["event"]
        chip_ids = batch["chip_id"]

        B = images.shape[0]
        aug_preds: list[torch.Tensor] = []

        for k in range(n_aug):
            x_aug  = tta_augment(images, k)              # (B, 6, H, W) augmented
            logits = model(x_aug)                         # (B, 1, H, W)
            probs  = torch.sigmoid(logits).squeeze(1)     # (B, H, W)
            probs  = tta_deaugment(probs, k)              # back to original orientation
            aug_preds.append(probs.cpu())

        stacked   = torch.stack(aug_preds, dim=0)   # (n_aug, B, H, W)
        mean_prob = stacked.mean(dim=0)              # (B, H, W)
        variance  = stacked.var(dim=0)               # (B, H, W)

        for i in range(B):
            results.append({
                "mean_prob": mean_prob[i].numpy(),
                "variance":  variance[i].numpy(),
                "label":     labels[i].numpy(),
                "event":     events[i],
                "chip_id":   chip_ids[i],
            })

    return results


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
# 1b. Test-Time Augmentation (TTA) inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_inference(
    model:    torch.nn.Module,
    loader:   torch.utils.data.DataLoader,
    device:   torch.device,
) -> list[dict]:
    """
    Test-Time Augmentation (TTA) inference using 8 geometric transforms.

    Applies the D4 symmetry group (4 rotations × 2 flips) to each chip,
    runs a single forward pass per augmentation, then averages predictions.
    The variance across augmented predictions provides a geometric uncertainty
    signal without requiring MC Dropout or model retraining.

    TTA is especially useful for Variants A/B/C (dropout_rate=0.0) where
    MC Dropout produces zero variance. TTA gives 5–15x larger variance signal
    that is correlated with model prediction difficulty.

    Returns the same format as mc_dropout_inference() for drop-in compatibility.
    """
    model.eval()

    results = []

    for batch in tqdm(loader, desc="TTA inference (8 augmentations)"):
        images   = batch["image"].to(device)   # (B, 6, H, W)
        labels   = batch["label"]
        events   = batch["event"]
        chip_ids = batch["chip_id"]

        B, C, H, W = images.shape
        tta_preds: list[torch.Tensor] = []

        # 8 augmentations: 4 rotations × 2 flips
        for k in range(4):              # 0°, 90°, 180°, 270°
            for flip in [False, True]:  # no flip, horizontal flip
                aug = torch.rot90(images, k=k, dims=(-2, -1))
                if flip:
                    aug = torch.flip(aug, dims=[-1])

                logits = model(aug)                         # (B, 1, H, W)
                probs  = torch.sigmoid(logits).squeeze(1)  # (B, H, W)

                # Invert augmentation on prediction
                if flip:
                    probs = torch.flip(probs, dims=[-1])
                probs = torch.rot90(probs, k=(4 - k) % 4, dims=(-2, -1))

                tta_preds.append(probs.cpu())

        tta_preds = torch.stack(tta_preds, dim=0)   # (8, B, H, W)
        mean_prob = tta_preds.mean(dim=0)            # (B, H, W)
        variance  = tta_preds.var(dim=0)             # (B, H, W)

        for i in range(B):
            results.append({
                "mean_prob": mean_prob[i].numpy(),
                "variance":  variance[i].numpy(),
                "label":     labels[i].numpy(),
                "event":     events[i],
                "chip_id":   chip_ids[i],
            })

    return results


# ─────────────────────────────────────────────────────────────
# 1c. Logit collection (single forward pass — for temperature scaling)
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
# 3b. Logit distribution diagnostics
# ─────────────────────────────────────────────────────────────

def analyze_logit_distribution(
    logits: np.ndarray,
    labels: np.ndarray,
    out_path: str | None = None,
    ignore_value: int = -1,
) -> dict:
    """
    Diagnostic analysis of raw model logit distribution.

    Helps explain extreme optimal temperatures (e.g. T=0.100 means the
    model is severely overconfident — logits span ±50 instead of ±3).

    For a well-calibrated binary classifier:
      - Flood logits    should cluster around +1 to +3
      - No-flood logits should cluster around −1 to −3
      - T ≈ 1.0 means calibration is good
      - T < 0.5 means the model is over-confident (logits too large)
      - T > 2.0 means the model is under-confident (logits too small)

    Args:
        logits:       (N,) or (H, W) raw model output (before sigmoid)
        labels:       same shape as logits, int (0/1/-1)
        out_path:     optional path to save histogram figure
        ignore_value: label value to exclude

    Returns:
        dict with keys: mean_flood, std_flood, mean_no_flood, std_no_flood,
                        p95_abs, fraction_extreme (|logit| > 10)
    """
    logits = logits.flatten().astype(np.float32)
    labels = labels.flatten()

    valid      = labels != ignore_value
    logits_v   = logits[valid]
    labels_v   = labels[valid]

    flood_mask    = labels_v == 1
    no_flood_mask = labels_v == 0

    flood_logits    = logits_v[flood_mask]
    no_flood_logits = logits_v[no_flood_mask]

    stats = {
        "n_flood":          int(flood_mask.sum()),
        "n_no_flood":       int(no_flood_mask.sum()),
        "mean_flood":       float(flood_logits.mean())    if flood_mask.any()    else float("nan"),
        "std_flood":        float(flood_logits.std())     if flood_mask.any()    else float("nan"),
        "mean_no_flood":    float(no_flood_logits.mean()) if no_flood_mask.any() else float("nan"),
        "std_no_flood":     float(no_flood_logits.std())  if no_flood_mask.any() else float("nan"),
        "p95_abs":          float(np.percentile(np.abs(logits_v), 95)),
        "fraction_extreme": float((np.abs(logits_v) > 10).mean()),
    }

    # Interpretation hint
    p95 = stats["p95_abs"]
    if p95 > 10:
        hint = (f"SEVERELY overconfident (95th‐pct |logit|={p95:.1f}). "
                f"Expected temperature T ≈ {p95/3:.2f} to re-calibrate to ±3 range.")
    elif p95 > 5:
        hint = (f"Moderately overconfident (95th‐pct |logit|={p95:.1f}). "
                f"Expected T ≈ {p95/3:.2f}.")
    else:
        hint = f"Logit range looks reasonable (95th‐pct |logit|={p95:.1f})."
    stats["interpretation"] = hint
    print(f"[analyze_logit_distribution] {hint}")

    if out_path is not None:
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(8, 4))
        if no_flood_mask.any():
            ax.hist(no_flood_logits, bins=100, alpha=0.6, label="No-flood (0)",
                    color="steelblue", density=True)
        if flood_mask.any():
            ax.hist(flood_logits, bins=100, alpha=0.6, label="Flood (1)",
                    color="firebrick", density=True)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Raw logit (pre-sigmoid)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Logit distribution — {hint[:60]}", fontsize=10)
        ax.legend(fontsize=10)
        _plt.tight_layout()
        _plt.savefig(out_path, dpi=150, bbox_inches="tight")
        _plt.close()
        print(f"  Saved: {out_path}")

    return stats


# ─────────────────────────────────────────────────────────────
# 3c. Temperature scaling
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

    # ── Logit distribution diagnostics (optional) ──────────────
    if getattr(args, "analyze_logits", False):
        print("Analyzing raw logit distribution (single deterministic pass)...")
        model.eval()
        all_logits: list[np.ndarray] = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Logit analysis"):
                imgs = batch["image"].to(device)
                lgs  = model(imgs).squeeze(1).cpu().numpy().flatten()
                all_logits.append(lgs)
        lgs_all = np.concatenate(all_logits)
        print(f"  Logit mean={lgs_all.mean():.4f}  std={lgs_all.std():.4f}")
        print(f"  Logit range=[{lgs_all.min():.3f}, {lgs_all.max():.3f}]")
        near_zero = np.mean(np.abs(lgs_all) < 0.5)
        print(f"  Fraction |logit| < 0.5 (uncertain region): {near_zero:.3%}")
        probs_diag = 1.0 / (1.0 + np.exp(-lgs_all))
        print(f"  Prob mean={probs_diag.mean():.4f}  "
              f"fraction > 0.9: {np.mean(probs_diag > 0.9):.3%}  "
              f"fraction < 0.1: {np.mean(probs_diag < 0.1):.3%}\n")
        del all_logits, lgs_all, probs_diag

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

    # ── Logit distribution diagnostic (optional) ────────────────
    if getattr(args, "analyze_logits", False):
        print("Analysing logit distribution on val set...")
        val_logits_diag, val_labels_diag = collect_logits(model, val_loader, device)
        analyze_logit_distribution(
            val_logits_diag, val_labels_diag,
            out_path=str(out_dir / "logit_distribution.png"),
        )

    # ── Inference: TTA or MC Dropout ───────────────────────────
    use_tta    = getattr(args, "use_tta", False)
    include_bn = getattr(args, "include_bn", False)

    if use_tta:
        n_aug = getattr(args, "n_aug", 8)
        print(f"Using TTA ({n_aug} D4 augmentations) — no dropout active\n")
        results = tta_inference(model, test_loader, device, n_aug=n_aug)
    else:
        print(f"Using MC Dropout (T={args.T} passes)\n")
        results = mc_dropout_inference(model, test_loader, device,
                                       T=args.T, include_bn=include_bn)
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

    # Optional: save mean_prob and variance as .npy for downstream tools
    if getattr(args, "save_arrays", False):
        arrays_dir = out_dir / "arrays"
        arrays_dir.mkdir(exist_ok=True)
        print(f"\nSaving .npy arrays to {arrays_dir}...")
        for r in results:
            cid = r["chip_id"]
            np.save(str(arrays_dir / f"chip_{cid}_mean.npy"), r["mean_prob"])
            np.save(str(arrays_dir / f"chip_{cid}_var.npy"),  r["variance"])
        print(f"  Saved {len(results)} chips × 2 arrays")

    print(f"\nDone. Results in {out_dir}")
    return results, metrics


# ─────────────────────────────────────────────────────────────
# 7.  Argument parser
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MC Dropout / TTA Uncertainty Estimation")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to best.pt checkpoint")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/uncertainty")
    p.add_argument("--T",            type=int, default=20,
                   help="Number of MC Dropout forward passes (ignored with --use_tta)")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--uncertainty_threshold", type=float, default=0.05,
                   help="Max variance for trusted pixel")
    p.add_argument("--n_maps",       type=int, default=10,
                   help="Number of uncertainty map figures to save")
    p.add_argument("--calibrate",    action="store_true",
                   help="Find optimal temperature T on val set and apply to test predictions")
    p.add_argument("--include_bn",   action="store_true",
                   help="Set BatchNorm2d layers to train mode during MC passes "
                        "(Teye et al. 2018). Uses batch statistics instead of running "
                        "statistics, providing ~10–100× larger predictive variance. "
                        "Requires batch_size >= 4 for stable BN statistics.")
    p.add_argument("--save_arrays", action="store_true",
                   help="Save per-chip mean_prob and variance as numpy arrays "
                        "({out_dir}/chip_{chip_id}_mean.npy and _var.npy). "
                        "Required by tools/exposure_tau_sweep.py.")
    p.add_argument("--use_tta",    action="store_true",
                   help="Use Test-Time Augmentation (D4 group: 4 rotations × 2 flips) "
                        "instead of MC Dropout for uncertainty estimation. "
                        "TTA variance is typically 5–50× larger than MC Dropout variance "
                        "for bimodal models (T ≪ 1 after calibration). "
                        "When set, --T and --include_bn are ignored.")
    p.add_argument("--n_aug",      type=int, default=8,
                   help="Number of D4 augmentations for TTA (default=8 = full group). "
                        "Use 4 for rotations-only (faster). Only used when --use_tta is set.")
    p.add_argument("--analyze_logits", action="store_true",
                   help="Save logit distribution histogram and print summary stats. "
                        "Useful for diagnosing bimodal/collapsed predictions.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_uncertainty(args)