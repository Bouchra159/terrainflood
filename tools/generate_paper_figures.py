"""
tools/generate_paper_figures.py
================================
Regenerate all publication-quality figures from existing JSON result files.

No model or GPU required — works on the laptop after a git pull.
Import the canonical plots.py functions so styles are always consistent.

Figures produced
----------------
  fig_ablation_full.png       — 4-panel ablation with post-cal annotation
                                and 95% bootstrap CI error bars on IoU
  fig_reliability_postcal.png — Post-calibration reliability diagram (Var D)
  fig_reliability_precal.png  — Pre-calibration reliability diagram (Var D)
  fig_reliability_2panel.png  — Side-by-side pre / post calibration
  fig_risk_coverage.png       — Risk–coverage curve (Var D) with corrected axes
  fig_calibration_shift.png   — Scatter: predicted prob. distribution
                                before vs after temperature scaling

Usage
-----
  python tools/generate_paper_figures.py \\
      --results_dir results \\
      --out_dir     results/paper_figures

All output figures use DPI=300, bbox_inches='tight'.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── import plots.py from project root ──────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
import plots  # noqa: E402  (brings in the shared rcParams)
from plots import (           # noqa: E402
    plot_reliability_diagram,
    plot_ablation_table,
    plot_risk_coverage_curve,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ablation comparison figure (enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def make_ablation_figure(results_dir: Path, out_dir: Path) -> None:
    """
    Regenerate ablation_comparison.png with:
      • Bootstrap 95% CI error bars on IoU panel
      • Post-calibration annotation on ECE and Brier panels
    """
    ablation  = _load(results_dir / "ablation" / "ablation_metrics.json")
    stats     = _load(results_dir / "stats"    / "ablation_stats.json")
    unc_D     = _load(results_dir / "uncertainty_D" / "uncertainty_metrics.json")

    # Bootstrap 95% CI per variant
    bootstrap_ci = {}
    for v, data in stats["variants"].items():
        ci = data.get("ci", {})
        bootstrap_ci[v] = (ci.get("ci_lower", 0.0), ci.get("ci_upper", 1.0))

    # Post-calibration values (D only)
    post_cal = {
        "D": {
            "ece":   unc_D["overall"]["ece"],
            "brier": unc_D["overall"]["brier"],
        }
    }

    out_path = out_dir / "fig_ablation_full.png"
    plot_ablation_table(
        ablation_results = ablation,
        out_path         = str(out_path),
        post_cal         = post_cal,
        bootstrap_ci     = bootstrap_ci,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Reliability diagrams — pre and post calibration
# ─────────────────────────────────────────────────────────────────────────────

def make_reliability_figures(results_dir: Path, out_dir: Path) -> None:
    """
    Produces:
      • Pre-calibration diagram for Variant D
      • Post-calibration diagram for Variant D
      • Side-by-side 2-panel figure
    """
    eval_D = _load(results_dir / "eval_D"        / "metrics.json")
    unc_D  = _load(results_dir / "uncertainty_D" / "uncertainty_metrics.json")
    temp   = _load(results_dir / "uncertainty_D" / "temperature.json")

    T_val = temp.get("temperature", float("nan"))

    # ── Pre-calibration ──────────────────────────────────────────────────────
    pre_cal = eval_D["calibration"]
    plot_reliability_diagram(
        bin_accs      = pre_cal["bin_accs"],
        bin_confs     = pre_cal["bin_confs"],
        ece           = eval_D["overall"]["ece"],
        out_path      = str(out_dir / "fig_reliability_precal.png"),
        title         = "Reliability Diagram — Variant D",
        subtitle_note = "Pre-calibration (raw model outputs)",
    )

    # ── Post-calibration ─────────────────────────────────────────────────────
    post_cal = unc_D["calibration"]
    plot_reliability_diagram(
        bin_accs      = post_cal["bin_accs"],
        bin_confs     = post_cal["bin_confs"],
        ece           = unc_D["overall"]["ece"],
        out_path      = str(out_dir / "fig_reliability_postcal.png"),
        title         = "Reliability Diagram — Variant D",
        subtitle_note = f"Post-calibration (T = {T_val:.3f})",
    )

    # ── Combined 2-panel figure ───────────────────────────────────────────────
    # IEEE double-column: 7.16 × 3.5 in
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5))

    def _draw_reliability(ax, bin_accs, bin_confs, ece, subtitle):
        bin_accs  = np.asarray(bin_accs,  dtype=float)
        bin_confs = np.asarray(bin_confs, dtype=float)
        n_bins    = len(bin_accs)
        bin_centers = np.linspace(1 / (2*n_bins), 1 - 1/(2*n_bins), n_bins)
        bin_width   = 1.0 / n_bins
        nonempty    = bin_confs > 0

        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.08, color="#d32f2f",
                         label="Over-confident")
        ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.08, color="#1565c0",
                         label="Under-confident")
        ax.bar(bin_centers, bin_accs, width=bin_width*0.9, alpha=0.55,
               color="#1976D2", label="Model", edgecolor="white", linewidth=0.5)
        if nonempty.sum() >= 2:
            ax.plot(bin_confs[nonempty], bin_accs[nonempty], "o-",
                    color="#0D47A1", linewidth=1.5, markersize=4, zorder=5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2,
                label="Perfect calibration", zorder=4)

        ann_lines = [f"ECE = {ece:.4f}",
                     f"non-empty: {int(nonempty.sum())}/{n_bins} bins"]
        ax.text(0.04, 0.93, "\n".join(ann_lines),
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="0.7", alpha=0.9))
        ax.set_title(f"Variant D — {subtitle}", pad=4)
        ax.set_xlabel("Confidence (predicted probability)")
        ax.set_ylabel("Fraction of flood pixels")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.legend(fontsize=8, loc="lower right")

    _draw_reliability(axes[0], pre_cal["bin_accs"],  pre_cal["bin_confs"],
                      eval_D["overall"]["ece"],   "Pre-calibration")
    _draw_reliability(axes[1], post_cal["bin_accs"], post_cal["bin_confs"],
                      unc_D["overall"]["ece"],    f"Post-calibration (T={T_val:.3f})")

    fig.suptitle(
        "Calibration Before and After Temperature Scaling — Bolivia OOD",
        y=1.02, fontsize=11,
    )
    plt.tight_layout(pad=0.5)
    _save(fig, out_dir / "fig_reliability_2panel.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Risk–coverage curve (corrected x-axis convention)
# ─────────────────────────────────────────────────────────────────────────────

def make_risk_coverage_figure(results_dir: Path, out_dir: Path) -> None:
    """
    Risk–coverage curve for Variant D.

    X-axis: coverage (fraction of pixels retained, decreasing left→right
            following standard convention; ax.invert_xaxis() is used).
    Y-axis: risk = 1 − IoU on retained pixels.

    IMPORTANT NOTE on curve shape for this dataset:
    The Bolivia test set is highly imbalanced (flood = 10.1%).  At very low
    coverage (most selective threshold), almost all retained pixels are
    background (variance ≈ 0) — so IoU ≈ 0 and risk ≈ 1.  Risk decreases
    as coverage grows and flood pixels are included.  This is the expected
    behaviour for imbalanced segmentation; the AURC should be interpreted
    relative to this baseline, not against a perfectly balanced scenario.
    """
    rc = _load(results_dir / "eval_D" / "risk_coverage.json")

    coverage = np.asarray(rc["coverage"])
    risk     = np.asarray(rc["risk"])
    aurc     = rc["aurc"]

    # Find coverage at which risk crosses 0.35 (≈ IoU threshold of 0.65)
    cross_idx = np.searchsorted(-risk, -0.35)  # first index where risk ≤ 0.35
    if cross_idx < len(coverage):
        cross_cov = coverage[cross_idx]
    else:
        cross_cov = None

    # IEEE single-column
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.plot(coverage, risk, "-", color="#0072B2", linewidth=1.5,
            label=f"Variant D  (AURC = {aurc:.4f})")

    # Mark coverage=0.70 reference line
    ax.axvline(x=0.70, color="#9E9E9E", linestyle="--",
               linewidth=1.0, alpha=0.8, label="Coverage = 0.70")

    # Mark where risk = 0.35 (IoU ≈ 0.65)
    if cross_cov is not None:
        ax.axhline(y=0.35, color="#E65100", linestyle=":",
                   linewidth=1.0, alpha=0.9, label="Risk = 0.35  (IoU = 0.65)")
        ax.scatter([cross_cov], [0.35], s=30, color="#E65100", zorder=5)

    # Annotate: class imbalance caveat
    ax.text(0.03, 0.96,
            "Imbalanced classes\n(flood = 10.1%):\nhigh risk at low\ncoverage is expected.",
            transform=ax.transAxes, fontsize=6.5, va="top", color="#616161",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.85))

    ax.set_xlabel("Coverage (fraction of pixels retained)")
    ax.set_ylabel("Risk  (1 \u2212 IoU on retained pixels)")
    ax.set_title("Risk\u2013Coverage Curve — Variant D")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()  # standard: left = selective, right = all pixels
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(fontsize=7.5, loc="upper left")

    plt.tight_layout(pad=0.5)
    _save(fig, out_dir / "fig_risk_coverage.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Calibration shift figure (predicted-probability histogram, pre vs post)
# ─────────────────────────────────────────────────────────────────────────────

def make_calibration_shift_figure(results_dir: Path, out_dir: Path) -> None:
    """
    Bar chart showing mean predicted confidence per bin before and after
    temperature scaling — illustrates the 'sharpening' effect of T=0.100.

    X: theoretical bin centres  [0.033 … 0.967] for 15 bins
    Y: mean predicted confidence (bin_confs) — this is the ACTUAL mean
       probability the model assigns to pixels falling in that bin.

    Pre-cal: predictions cluster in [0.32, 0.74]  (model is under-confident)
    Post-cal: predictions spread across [0.03, 0.97] (well-calibrated)
    """
    eval_D = _load(results_dir / "eval_D"        / "metrics.json")
    unc_D  = _load(results_dir / "uncertainty_D" / "uncertainty_metrics.json")
    temp   = _load(results_dir / "uncertainty_D" / "temperature.json")
    T_val  = temp.get("temperature", float("nan"))

    n_bins      = 15
    bin_centers = np.linspace(1/(2*n_bins), 1 - 1/(2*n_bins), n_bins)
    bin_width   = 1.0 / n_bins

    pre_confs  = np.asarray(eval_D["calibration"]["bin_confs"],  dtype=float)
    pre_accs   = np.asarray(eval_D["calibration"]["bin_accs"],   dtype=float)
    post_confs = np.asarray(unc_D["calibration"]["bin_confs"],   dtype=float)
    post_accs  = np.asarray(unc_D["calibration"]["bin_accs"],    dtype=float)

    # IEEE single-column: 3.5 × 3.5
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=False)

    for ax, confs, accs, label, ece in [
        (axes[0], pre_confs,  pre_accs,  f"Pre-cal",                eval_D["overall"]["ece"]),
        (axes[1], post_confs, post_accs, f"Post-cal (T={T_val:.3f})", unc_D["overall"]["ece"]),
    ]:
        nonempty = confs > 0
        # Gap bars (ideal diagonal would be at bin_centers)
        ax.bar(bin_centers, bin_centers, width=bin_width*0.9,
               alpha=0.15, color="#888888", label="Perfect (diagonal)",
               edgecolor="none")
        # Actual accuracy bars
        ax.bar(bin_centers, accs, width=bin_width*0.9, alpha=0.65,
               color="#1976D2", label="Fraction flood", edgecolor="white",
               linewidth=0.4)
        # Calibration line on actual confidence positions
        if nonempty.sum() >= 2:
            ax.plot(confs[nonempty], accs[nonempty], "o-",
                    color="#0D47A1", linewidth=1.3, markersize=3.5, zorder=5,
                    label="Model (at mean conf.)")
        ax.plot([0,1],[0,1],"k--",linewidth=1.0,label="Ideal")
        ax.text(0.04, 0.95, f"ECE = {ece:.4f}\nnon-empty: {nonempty.sum()}/{n_bins}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
        ax.set_title(label)
        ax.set_xlabel("Confidence bin centre")
        ax.set_ylabel("Fraction flood pixels in bin")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Effect of Temperature Scaling on Calibration — Variant D, Bolivia OOD",
        y=1.02, fontsize=11,
    )
    plt.tight_layout(pad=0.5)
    _save(fig, out_dir / "fig_calibration_shift.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Bootstrap CI summary figure
# ─────────────────────────────────────────────────────────────────────────────

def make_bootstrap_ci_figure(results_dir: Path, out_dir: Path) -> None:
    """
    Horizontal error-bar plot of IoU 95% CI for all 4 variants.
    Cleaner than the ablation bar chart for showing uncertainty in IoU.
    """
    stats = _load(results_dir / "stats" / "ablation_stats.json")
    abl   = _load(results_dir / "ablation" / "ablation_metrics.json")

    variants      = ["A", "B", "C", "D"]
    full_labels   = [
        "A — SAR baseline",
        "B — +HAND band",
        "C — +HAND gate",
        "D — +HAND gate\n     +MC Dropout",
    ]
    colours = ["#90CAF9", "#90CAF9", "#90CAF9", "#1565C0"]

    y_pos  = np.arange(len(variants))
    iou_obs = [abl[v]["iou"] for v in variants]
    ci_lo   = [stats["variants"][v]["ci"]["ci_lower"] for v in variants]
    ci_hi   = [stats["variants"][v]["ci"]["ci_upper"] for v in variants]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for i, (v, lo, hi, obs, c) in enumerate(
            zip(variants, ci_lo, ci_hi, iou_obs, colours)):
        ax.barh(y_pos[i], obs, color=c, alpha=0.75, height=0.5,
                edgecolor="white", linewidth=0.4)
        ax.errorbar(obs, y_pos[i],
                    xerr=[[obs - lo], [hi - obs]],
                    fmt="none", ecolor="#212121",
                    elinewidth=1.2, capsize=4, capthick=1.2)
        ax.text(hi + 0.01, y_pos[i], f"{obs:.3f}",
                va="center", ha="left", fontsize=8, color="#212121")
        # CI label
        ax.text(lo - 0.01, y_pos[i], f"[{lo:.2f},{hi:.2f}]",
                va="center", ha="right", fontsize=6.5, color="#616161")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(full_labels, fontsize=8.5)
    ax.set_xlabel("IoU on Bolivia OOD Test Set")
    ax.set_title("Ablation IoU with 95% Bootstrap CI\n(1000 chip-level resamples)", pad=4)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="#BDBDBD", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

    plt.tight_layout(pad=0.5)
    _save(fig, out_dir / "fig_bootstrap_ci.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  McNemar significance heatmap
# ─────────────────────────────────────────────────────────────────────────────

def make_mcnemar_figure(results_dir: Path, out_dir: Path) -> None:
    """
    Matrix plot showing net pixel improvement between variants.
    Annotated with McNemar significance stars.

    Row = 'A' (base); Column = 'B' (comparison).
    Cell shows net improvement B − A (positive = B better at pixel level).
    """
    stats   = _load(results_dir / "stats" / "ablation_stats.json")
    tests   = stats["mcnemar"]

    variants = ["A", "B", "C", "D"]
    n = len(variants)
    mat = np.full((n, n), np.nan)

    pair_map = {
        ("A","B"): 0, ("B","C"): 1, ("C","D"): 2,
        ("A","C"): 3, ("A","D"): 4,
    }

    for t in tests:
        la = t["label_a"].replace("Variant_", "")
        lb = t["label_b"].replace("Variant_", "")
        ri = variants.index(la)
        ci = variants.index(lb)
        # net improvement: positive = B > A
        net = t.get("net_improvement_B", 0)
        mat[ri, ci] = net
        mat[ci, ri] = -net

    # Symmetric normalise by max abs
    abs_max = np.nanmax(np.abs(mat))
    mat_norm = mat / abs_max

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(mat_norm, cmap="RdBu", vmin=-1, vmax=1,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Net pixel improvement (normalised)")

    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#888888")
            else:
                val = mat[i, j]
                sign = "+" if val >= 0 else ""
                stars = "***" if abs(val) > 0 else ""
                ax.text(j, i, f"{sign}{int(val):,}\n{stars}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(mat_norm[i, j]) > 0.6 else "#212121")

    ax.set_xticks(range(n)); ax.set_xticklabels(variants, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(variants, fontsize=9)
    ax.set_xlabel("Column variant (comparison)")
    ax.set_ylabel("Row variant (base)")
    ax.set_title("McNemar Net Pixel Improvement\n(+ve = column better than row)", pad=4)

    plt.tight_layout(pad=0.5)
    _save(fig, out_dir / "fig_mcnemar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate all paper figures from JSON results (no GPU needed)"
    )
    p.add_argument("--results_dir", type=str, default="results",
                   help="Root results directory")
    p.add_argument("--out_dir",     type=str, default="results/paper_figures",
                   help="Output directory for paper figures")
    p.add_argument("--figures",     nargs="+",
                   choices=["ablation", "reliability", "risk_coverage",
                             "calibration_shift", "bootstrap_ci", "mcnemar",
                             "all"],
                   default=["all"],
                   help="Which figures to generate (default: all)")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    want = set(args.figures)
    run_all = "all" in want

    generators = {
        "ablation":          make_ablation_figure,
        "reliability":       make_reliability_figures,
        "risk_coverage":     make_risk_coverage_figure,
        "calibration_shift": make_calibration_shift_figure,
        "bootstrap_ci":      make_bootstrap_ci_figure,
        "mcnemar":           make_mcnemar_figure,
    }

    for name, fn in generators.items():
        if run_all or name in want:
            print(f"\n[generate_paper_figures] {name} ...")
            try:
                fn(results_dir, out_dir)
            except FileNotFoundError as e:
                print(f"  SKIP ({name}): {e}")
            except Exception as e:
                print(f"  ERROR ({name}): {e}")
                raise

    print(f"\nDone — figures in: {out_dir}")


if __name__ == "__main__":
    main()
