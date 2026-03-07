"""
Phase 5 — Population Exposure
================================
File: 06_exposure.py

Uncertainty-gated population exposure estimation.

Combines flood probability maps from MC Dropout with WorldPop density
to estimate people at risk — with honest 90% confidence bounds.

Two exposure estimates per chip:
  deterministic_exposure : sum(pop × P(flood))  over all valid pixels
  gated_exposure         : same, but only inside the trust mask
  uncertain_exposure     : exposure in uncertain zones (for planners)

90% CI from MC samples:
  For each MC pass t: exposure_t = sum(pop × prob_t) inside trust mask
  CI = [5th percentile, 95th percentile] across T passes

Usage:
  python 06_exposure.py \\
      --checkpoint checkpoints/variant_D_TIMESTAMP/best.pt \\
      --data_root  data/sen1floods11 \\
      --pop_dir    data/pop_chips \\
      --output_dir results/exposure \\
      --T 20
"""

import json
import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling

from trust_mask import compute_trust_mask, summarise_trust_mask


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

from model   import build_model     # noqa: E402
from dataset import get_dataloaders  # noqa: E402

# Import mc_dropout_inference from 05_uncertainty.py
_unc = _import_module("uncertainty_mod", str(_root / "05_uncertainty.py"))
mc_dropout_inference = _unc.mc_dropout_inference


# ─────────────────────────────────────────────────────────────
# 1.  WorldPop loading
# ─────────────────────────────────────────────────────────────

def load_worldpop(
    pop_path:     str | Path,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    Reads a WorldPop GeoTIFF and resamples to target_shape.

    WorldPop is at 100 m resolution; chips are at 10 m.
    We use bilinear resampling and treat values as people per pixel.

    Args:
        pop_path:     path to a *_pop.tif file
        target_shape: (H, W) pixel dimensions of the flood chip

    Returns:
        pop: (H, W) float32 array — population count per pixel.
             Returns zeros if file not found (graceful degradation).
    """
    pop_path = Path(pop_path)
    if not pop_path.exists():
        return np.zeros(target_shape, dtype=np.float32)

    with rasterio.open(pop_path) as src:
        data = src.read(
            1,
            out_shape=(target_shape[0], target_shape[1]),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0, None)
    return data


# ─────────────────────────────────────────────────────────────
# 2.  Single-chip exposure
# ─────────────────────────────────────────────────────────────

def compute_exposure(
    flood_prob:   np.ndarray,
    trust_mask:   np.ndarray,
    population:   np.ndarray,
    label:        np.ndarray | None = None,
    ignore_value: int = -1,
) -> dict:
    """
    Expected population exposure from a single flood probability map.

    Args:
        flood_prob:  (H, W) predicted flood probability in [0, 1]
        trust_mask:  (H, W) bool — True = trusted pixel
        population:  (H, W) people per pixel (from WorldPop)
        label:       optional (H, W) ground truth
        ignore_value: label value to ignore

    Returns:
        dict with:
          deterministic_exposure : sum(pop × P) over all valid pixels
          gated_exposure         : sum(pop × P) inside trust mask only
          uncertain_exposure     : difference (untrusted zone)
          coverage               : fraction of valid pixels that are trusted
    """
    if label is not None:
        valid = label != ignore_value
    else:
        valid = np.ones_like(flood_prob, dtype=bool)

    det_exposure   = float((population[valid] * flood_prob[valid]).sum())
    trusted_valid  = valid & trust_mask
    gated_exposure = float((population[trusted_valid] * flood_prob[trusted_valid]).sum())
    uncertain_exp  = det_exposure - gated_exposure

    trust_stats = summarise_trust_mask(trust_mask, label)

    return {
        "deterministic_exposure": round(det_exposure,   1),
        "gated_exposure":         round(gated_exposure, 1),
        "uncertain_exposure":     round(uncertain_exp,  1),
        "coverage":               round(trust_stats["coverage"], 4),
        "trusted_pixels":         trust_stats["trusted_pixels"],
        "total_pixels":           trust_stats["total_pixels"],
    }


def compute_exposure_ci(
    mc_probs:     np.ndarray,
    trust_mask:   np.ndarray,
    population:   np.ndarray,
    label:        np.ndarray | None = None,
    ci_lower:     float = 5.0,
    ci_upper:     float = 95.0,
    ignore_value: int   = -1,
) -> dict:
    """
    90% confidence interval for population exposure from MC samples.

    For each forward pass t: compute gated exposure inside trust mask.
    The distribution of T exposure values gives the CI.

    Args:
        mc_probs:  (T, H, W) per-pass flood probabilities
        trust_mask:(H, W) bool derived from variance of mc_probs
        population:(H, W) people per pixel
        label:     optional (H, W) ground truth
        ci_lower:  lower percentile (default 5)
        ci_upper:  upper percentile (default 95 → 90% CI)

    Returns:
        dict with mean_exposure, ci_lower, ci_upper, ci_width, std_exposure
    """
    T = mc_probs.shape[0]

    if label is not None:
        valid = label != ignore_value
    else:
        valid = np.ones(mc_probs.shape[1:], dtype=bool)

    trusted_valid = valid & trust_mask

    exposures = np.array([
        float((population[trusted_valid] * mc_probs[t][trusted_valid]).sum())
        for t in range(T)
    ])

    return {
        "mean_exposure": round(float(exposures.mean()), 1),
        "ci_lower":      round(float(np.percentile(exposures, ci_lower)), 1),
        "ci_upper":      round(float(np.percentile(exposures, ci_upper)), 1),
        "ci_width":      round(float(np.percentile(exposures, ci_upper) -
                                     np.percentile(exposures, ci_lower)), 1),
        "std_exposure":  round(float(exposures.std()), 1),
        "n_mc_samples":  T,
    }


# ─────────────────────────────────────────────────────────────
# 3.  Batch run
# ─────────────────────────────────────────────────────────────

def run_exposure_analysis(
    results:               list,
    pop_dir:               str | Path,
    out_path:              str | Path,
    ci_lower:              float = 5.0,
    ci_upper:              float = 95.0,
    uncertainty_threshold: float = 0.05,
) -> dict:
    """
    Runs exposure analysis over all chips from mc_dropout_inference().

    Args:
        results:   list of per-chip dicts from mc_dropout_inference()
                   Expected keys: mean_prob, variance, label, event, chip_id
                   Optional key: mc_passes (T, H, W) for CI computation
        pop_dir:   directory containing {chip_id}_pop.tif files
        out_path:  path to write exposure_results.json
        ci_lower:  lower percentile for CI
        ci_upper:  upper percentile for CI
        uncertainty_threshold: variance threshold for trust mask

    Returns:
        nested dict: overall / per_event / per_chip exposure estimates
    """
    pop_dir    = Path(pop_dir)
    event_data = defaultdict(list)
    per_chip   = []

    for r in results:
        chip_id    = r["chip_id"]
        event_name = r["event"]
        mean_prob  = r["mean_prob"]   # (H, W)
        variance   = r["variance"]    # (H, W)
        label      = r["label"]       # (H, W)

        trust_mask = compute_trust_mask(variance, uncertainty_threshold)

        H, W       = mean_prob.shape
        pop_path   = pop_dir / f"{chip_id}_pop.tif"
        population = load_worldpop(pop_path, target_shape=(H, W))

        chip_exp = compute_exposure(mean_prob, trust_mask, population, label)
        chip_exp.update({"chip_id": chip_id, "event": event_name})

        # CI from MC passes if available
        mc_passes = r.get("mc_passes")   # (T, H, W) or None
        if mc_passes is not None:
            ci = compute_exposure_ci(
                mc_passes, trust_mask, population, label,
                ci_lower=ci_lower, ci_upper=ci_upper,
            )
            chip_exp.update(ci)

        per_chip.append(chip_exp)
        event_data[event_name].append(chip_exp)

    # Per-event aggregation
    per_event    = {}
    total_gated  = total_det = total_uncert = 0.0

    for event_name, chips in event_data.items():
        ev_gated  = sum(c["gated_exposure"]          for c in chips)
        ev_det    = sum(c["deterministic_exposure"]   for c in chips)
        ev_uncert = sum(c["uncertain_exposure"]        for c in chips)
        avg_cov   = float(np.mean([c["coverage"] for c in chips]))

        per_event[event_name] = {
            "gated_exposure":         round(ev_gated,  1),
            "deterministic_exposure": round(ev_det,    1),
            "uncertain_exposure":     round(ev_uncert, 1),
            "mean_coverage":          round(avg_cov, 4),
            "n_chips":                len(chips),
        }
        total_gated  += ev_gated
        total_det    += ev_det
        total_uncert += ev_uncert

    output = {
        "overall": {
            "gated_exposure":         round(total_gated,  1),
            "deterministic_exposure": round(total_det,    1),
            "uncertain_exposure":     round(total_uncert, 1),
            "uncertainty_threshold":  uncertainty_threshold,
        },
        "per_event": per_event,
        "per_chip":  per_chip,
    }

    Path(out_path).write_text(json.dumps(output, indent=2))
    print(f"Exposure results → {out_path}")
    return output


# ─────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt    = torch.load(args.checkpoint, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"Loaded Variant {variant}  epoch={ckpt['epoch']}  "
          f"best_iou={ckpt['best_iou']:.4f}")

    # Test loader (Bolivia OOD)
    _, _, test_loader = get_dataloaders(
        data_root   = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # MC Dropout inference
    results = mc_dropout_inference(model, test_loader, device, T=args.T)
    print(f"Processed {len(results)} chips")

    # Exposure analysis
    out_path = out_dir / "exposure_results.json"
    output   = run_exposure_analysis(
        results               = results,
        pop_dir               = args.pop_dir,
        out_path              = out_path,
        uncertainty_threshold = args.uncertainty_threshold,
    )

    overall = output["overall"]
    print(f"\n{'='*50}")
    print(f"Gated exposure     : {overall['gated_exposure']:>12,.0f} people")
    print(f"Deterministic      : {overall['deterministic_exposure']:>12,.0f} people")
    print(f"Uncertain zone     : {overall['uncertain_exposure']:>12,.0f} people")
    print(f"{'='*50}")
    print(f"\nResults → {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uncertainty-gated population exposure")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to best.pt checkpoint")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--pop_dir",      type=str, default="data/pop_chips")
    p.add_argument("--output_dir",   type=str, default="results/exposure")
    p.add_argument("--T",            type=int, default=20,
                   help="Number of MC Dropout forward passes")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--uncertainty_threshold", type=float, default=0.05,
                   help="Max variance for trusted pixel")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
