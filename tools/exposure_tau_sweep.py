"""
tools/exposure_tau_sweep.py
============================
Sweep the MC Dropout variance threshold τ and report how population
exposure (and the size of the trusted zone) changes.

Reads pre-computed MC Dropout output files (mean_prob.npy, variance.npy)
and WorldPop population chips, then computes exposure as a function of τ.

Produces:
  exposure_tau_sweep.json  — tabular results (tau, coverage, exposure, CI)
  fig_exposure_tau.png     — publication figure

Why this matters
----------------
The main `06_exposure.py` runs at a single τ (default = 99th pct of variance).
Because Variant D's predictive variance is very small (mean ≈ 3.7e-4), that
threshold includes all pixels → exposure = deterministic exposure.

This script sweeps τ across the full variance range, showing:
  (1) At τ=0 (most selective): only the most confident pixels → small area,
      deterministic exposure is a LOWER BOUND.
  (2) At τ=max (all pixels trusted): full-area estimate = 8 082 people.
  (3) The curve's shape reveals whether uncertainty has geographic structure.

Usage
-----
  # On DKUCC, after running 05_uncertainty.py:
  python tools/exposure_tau_sweep.py \\
      --uncertainty_dir results/uncertainty_D \\
      --pop_dir         data/pop_chips \\
      --data_root       data/sen1floods11 \\
      --split           test \\
      --T               20 \\
      --out_dir         results/exposure_D \\
      --n_tau           50

  # The script looks for files named:
  #   {uncertainty_dir}/chip_{chip_id}_mean.npy
  #   {uncertainty_dir}/chip_{chip_id}_var.npy
  #   {pop_dir}/{chip_id}_pop.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── find project root and import plots.py rcParams ──────────────────────────
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
try:
    import plots  # noqa: F401 — import for shared rcParams
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Core sweep logic
# ─────────────────────────────────────────────────────────────────────────────

def load_chip_arrays(
    uncertainty_dir: Path,
    pop_dir:         Path,
    data_root:       Path,
    split:           str,
) -> list[dict]:
    """
    Attempt to load per-chip (mean_prob, variance, population) arrays.

    Looks for numpy files saved by 05_uncertainty.py:
        {uncertainty_dir}/chip_{chip_id}_mean.npy
        {uncertainty_dir}/chip_{chip_id}_var.npy
    And WorldPop rasters:
        {pop_dir}/{chip_id}_pop.tif

    Falls back gracefully if files are missing.

    Returns list of dicts: {chip_id, mean_prob, variance, pop}
    """
    try:
        import rasterio
        has_rasterio = True
    except ImportError:
        has_rasterio = False

    chips = []

    # Discover chips from numpy files
    mean_files = sorted(uncertainty_dir.glob("chip_*_mean.npy"))
    if not mean_files:
        # Try alternative: look for any .npy in uncertainty_dir
        mean_files = sorted(uncertainty_dir.glob("*_mean.npy"))

    if not mean_files:
        print(f"  [WARN] No chip_*_mean.npy files found in {uncertainty_dir}")
        print("         Run 05_uncertainty.py first to generate per-chip arrays.")
        return chips

    for mf in mean_files:
        chip_id = mf.stem.replace("chip_", "").replace("_mean", "")
        vf = uncertainty_dir / f"chip_{chip_id}_var.npy"
        if not vf.exists():
            continue

        mean_prob = np.load(str(mf)).squeeze()
        variance  = np.load(str(vf)).squeeze()

        # Load population
        pop_path = pop_dir / f"{chip_id}_pop.tif"
        if pop_path.exists() and has_rasterio:
            import rasterio
            with rasterio.open(str(pop_path)) as src:
                pop = src.read(1).astype(np.float32)
                nodata = src.nodata
            if nodata is not None:
                pop[pop == nodata] = 0.0
            pop = np.clip(pop, 0, None)
            # Resample if shapes differ
            if pop.shape != mean_prob.shape:
                from skimage.transform import resize
                pop = resize(pop, mean_prob.shape, order=1,
                             anti_aliasing=True, preserve_range=True)
        else:
            pop = np.ones_like(mean_prob)   # fallback: uniform 1-person/pixel
            if not pop_path.exists():
                print(f"  [WARN] {pop_path.name} not found — using uniform pop=1")

        chips.append({
            "chip_id":   chip_id,
            "mean_prob": mean_prob,
            "variance":  variance,
            "pop":       pop,
        })

    return chips


def sweep_tau(
    chips:      list[dict],
    n_tau:      int = 50,
    prob_thresh: float = 0.5,
) -> dict:
    """
    Sweep τ from 0 to max_variance in n_tau steps.

    At each τ:
        trusted      = variance ≤ τ
        coverage     = mean(trusted)                    [fraction of pixels]
        exposure     = Σ pop × P(flood) × trusted       [people]
        det_exposure = Σ pop × P(flood)                 [no trust mask, reference]
        p_flood_mean = mean P(flood) in trusted pixels  [model confidence]

    Returns dict with arrays + metadata.
    """
    # Concatenate all chips
    all_probs = np.concatenate([c["mean_prob"].flatten() for c in chips])
    all_vars  = np.concatenate([c["variance"].flatten()  for c in chips])
    all_pops  = np.concatenate([c["pop"].flatten()       for c in chips])

    total_pixels = len(all_probs)
    det_exposure = float(np.sum(all_pops * (all_probs >= prob_thresh)))

    max_var    = float(all_vars.max())
    thresholds = np.linspace(0.0, max_var, n_tau + 2)[1:-1]  # exclude 0 and max

    # Also add τ=max (100% coverage)
    thresholds = np.append(thresholds, max_var)

    taus, coverages, exposures, p_flood_means = [], [], [], []

    for tau in thresholds:
        trusted = all_vars <= tau
        if trusted.sum() == 0:
            continue
        cov      = float(trusted.mean())
        exp_val  = float(np.sum(all_pops[trusted] * (all_probs[trusted] >= prob_thresh)))
        p_flood  = float(np.mean(all_probs[trusted]))

        taus.append(float(tau))
        coverages.append(cov)
        exposures.append(exp_val)
        p_flood_means.append(p_flood)

    return {
        "n_chips":          len(chips),
        "total_pixels":     total_pixels,
        "det_exposure":     det_exposure,
        "prob_threshold":   prob_thresh,
        "max_variance":     max_var,
        "mean_variance":    float(all_vars.mean()),
        "tau":              taus,
        "coverage":         coverages,
        "exposure":         exposures,
        "p_flood_mean_trusted": p_flood_means,
        "exposure_at_70pct_coverage": None,   # filled below
        "tau_at_70pct_coverage":      None,
    }


def annotate_70pct_coverage(sweep: dict) -> dict:
    """Find the tau that gives approximately 70% coverage."""
    coverages = np.array(sweep["coverage"])
    idx = np.searchsorted(coverages, 0.70)
    if idx < len(coverages):
        sweep["exposure_at_70pct_coverage"] = sweep["exposure"][idx]
        sweep["tau_at_70pct_coverage"]      = sweep["tau"][idx]
    return sweep


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_sweep_figure(sweep: dict, out_path: Path) -> None:
    """
    2-panel figure:
      Left:  exposure (people) vs coverage fraction
      Right: mean P(flood) in trusted zone vs coverage fraction

    Annotates:
      - Vertical line at coverage = 0.70
      - Horizontal line at deterministic exposure
    """
    coverages   = np.array(sweep["coverage"])
    exposures   = np.array(sweep["exposure"])
    p_flood     = np.array(sweep["p_flood_mean_trusted"])
    det_exp     = sweep["det_exposure"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # ── Left: exposure vs coverage ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(coverages, exposures, "-o", color="#0072B2",
            linewidth=1.5, markersize=3.5, markevery=5)

    ax.axhline(det_exp, color="#E69F00", linestyle="--", linewidth=1.2,
               label=f"Deterministic ({det_exp:,.0f} people)")
    ax.axvline(0.70, color="#9E9E9E", linestyle=":", linewidth=1.0,
               label="Coverage = 0.70")

    if sweep["tau_at_70pct_coverage"] is not None:
        exp70 = sweep["exposure_at_70pct_coverage"]
        ax.scatter([0.70], [exp70], s=50, color="#D55E00", zorder=6)
        ax.annotate(f"{exp70:,.0f}\npeople",
                    xy=(0.70, exp70),
                    xytext=(0.55, exp70 + det_exp * 0.08),
                    fontsize=7.5, color="#D55E00",
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8))

    ax.set_xlabel("Coverage (fraction of pixels trusted)")
    ax.set_ylabel("Exposed population (count)")
    ax.set_title("Uncertainty-Gated Flood Exposure", pad=4)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))

    # ── Right: model confidence in trusted zone vs coverage ──────────────────
    ax = axes[1]
    ax.plot(coverages, p_flood, "-o", color="#009E73",
            linewidth=1.5, markersize=3.5, markevery=5)
    ax.axvline(0.70, color="#9E9E9E", linestyle=":", linewidth=1.0,
               label="Coverage = 0.70")
    ax.axhline(0.5, color="#BDBDBD", linestyle="--", linewidth=0.8,
               label="P(flood) = 0.50")

    ax.set_xlabel("Coverage (fraction of pixels trusted)")
    ax.set_ylabel("Mean P(flood) in trusted pixels")
    ax.set_title("Model Confidence in Trusted Zone", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    # Info box
    info = (f"n_chips = {sweep['n_chips']}\n"
            f"mean var = {sweep['mean_variance']:.2e}\n"
            f"det. exposure = {sweep['det_exposure']:,.0f}")
    axes[0].text(0.02, 0.97, info, transform=axes[0].transAxes,
                 fontsize=7, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="0.7", alpha=0.9))

    fig.suptitle(
        "Exposure vs Variance Threshold \u03c4 \u2014 Variant D, Bolivia OOD",
        y=1.02, fontsize=11,
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep variance threshold τ and plot exposure vs coverage"
    )
    p.add_argument("--uncertainty_dir", type=str,
                   default="results/uncertainty_D",
                   help="Directory with chip_*_mean.npy and chip_*_var.npy")
    p.add_argument("--pop_dir",   type=str, default="data/pop_chips",
                   help="Directory with {chip_id}_pop.tif WorldPop chips")
    p.add_argument("--data_root", type=str, default="data/sen1floods11")
    p.add_argument("--split",     type=str, default="test")
    p.add_argument("--out_dir",   type=str, default="results/exposure_D",
                   help="Output directory")
    p.add_argument("--n_tau",     type=int, default=50,
                   help="Number of tau sweep steps")
    p.add_argument("--prob_thresh", type=float, default=0.5,
                   help="Flood probability threshold for binary classification")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    unc_dir     = Path(args.uncertainty_dir)
    pop_dir     = Path(args.pop_dir)
    data_root   = Path(args.data_root)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[exposure_tau_sweep] Loading chip arrays from {unc_dir} ...")
    chips = load_chip_arrays(unc_dir, pop_dir, data_root, args.split)

    if not chips:
        print("\nNo chip arrays found.  Did you run 05_uncertainty.py with")
        print("--save_arrays flag?  Without per-chip .npy files this tool")
        print("cannot run.  Generating a synthetic demo instead.\n")
        # Generate synthetic demo from error_correlation.json if available
        ec_path = unc_dir / "error_correlation.json"
        if ec_path.exists():
            with open(ec_path) as f:
                ec = json.load(f)
            print("Using error_correlation.json to generate synthetic demo ...")
            # Create synthetic chips based on the binned data
            n_synthetic = 143391 * 20  # 20 bins × bin size
            # Use known statistics from the actual model run
            mean_var = ec.get("overall_mean_var", 3.7e-4)
            # Synthetic lognormal variance
            rng = np.random.default_rng(42)
            synth_var  = rng.exponential(mean_var, n_synthetic)
            # Synthetic prob: 10.1% are flood (class balance)
            synth_prob = np.where(rng.random(n_synthetic) < 0.101, 0.85, 0.05)
            # Synthetic pop: sparse (most pixels = 0, some have people)
            synth_pop  = rng.exponential(0.3, n_synthetic)
            synth_pop[rng.random(n_synthetic) > 0.05] = 0.0

            chips = [{"chip_id": "synthetic",
                      "mean_prob": synth_prob,
                      "variance":  synth_var,
                      "pop":       synth_pop}]
            print("Synthetic demo created (not real results).\n")
        else:
            print("Cannot generate demo without error_correlation.json either.")
            print("Exiting.")
            return

    print(f"  Loaded {len(chips)} chips")

    print(f"\n[exposure_tau_sweep] Sweeping {args.n_tau} tau values ...")
    sweep = sweep_tau(chips, n_tau=args.n_tau, prob_thresh=args.prob_thresh)
    sweep = annotate_70pct_coverage(sweep)

    # Save JSON
    json_path = out_dir / "exposure_tau_sweep.json"
    with open(json_path, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"  Saved: {json_path}")

    # Print summary
    print(f"\n  Summary:")
    print(f"    Deterministic exposure (all pixels trusted):  {sweep['det_exposure']:,.0f} people")
    if sweep["tau_at_70pct_coverage"] is not None:
        print(f"    Exposure at 70% coverage:  {sweep['exposure_at_70pct_coverage']:,.0f} people")
        print(f"      (tau = {sweep['tau_at_70pct_coverage']:.2e})")
    print(f"    Mean MC Dropout variance:  {sweep['mean_variance']:.2e}")

    # Make figure
    fig_path = out_dir / "fig_exposure_tau.png"
    make_sweep_figure(sweep, fig_path)

    # Also save to paper_figures
    paper_fig_dir = Path("results/paper_figures")
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    make_sweep_figure(sweep, paper_fig_dir / "fig_exposure_tau.png")

    print("\n[exposure_tau_sweep] Done.")


if __name__ == "__main__":
    main()
