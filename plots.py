"""
Plotting utilities for TerrainFlood-UQ thesis.
File: plots.py

All plots save to disk (Agg backend, no display).
Called by eval.py and uncertainty.py.

Functions:
  plot_flood_map()            — 4-panel chip figure (SAR / prob / variance / GT)
  plot_reliability_diagram()  — calibration curve with ECE annotation
  plot_coverage_accuracy()    — coverage-accuracy tradeoff curve
  plot_iou_bar_chart()        — per-event IoU bar chart
  plot_ablation_table()       — side-by-side variant comparison figure
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────
# Publication-quality style  (IEEE TGRS / RSE / IGARSS)
# Applied once at import; matches the journal body font and DPI
# requirements without requiring a separate style file.
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    # Resolution
    "figure.dpi":          160,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    # Fonts — serif for full journal; switch to "sans-serif" for conference
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif",
                            "Palatino", "Georgia", "serif"],
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.titleweight":    "bold",
    "axes.labelsize":      10,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "figure.titlesize":    12,
    "figure.titleweight":  "bold",
    # Math text matches LaTeX Computer Modern
    "mathtext.fontset":    "stix",
    # Lines
    "lines.linewidth":     1.5,
    "lines.markersize":    5,
    "patch.linewidth":     0.8,
    # Axes
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    # Grid
    "axes.grid":           True,
    "grid.alpha":          0.20,
    "grid.linestyle":      "--",
    "grid.linewidth":      0.5,
    "grid.color":          "#888888",
    # Legend
    "legend.framealpha":   0.85,
    "legend.edgecolor":    "0.7",
    "legend.borderpad":    0.4,
    # Ticks — inward, clean
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    3.5,
    "ytick.major.size":    3.5,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
})


# ─────────────────────────────────────────────────────────────
# 1.  Flood map — 4-panel chip figure
# ─────────────────────────────────────────────────────────────

def plot_flood_map(
    mean_prob:  np.ndarray,
    variance:   np.ndarray,
    trust_mask: np.ndarray,
    label:      np.ndarray,
    chip_id:    str,
    out_path:   str,
) -> None:
    """
    4-panel figure per chip:
      [Flood probability | Predictive variance | Trust mask | Ground truth]

    Args:
        mean_prob:  (H, W) predictive mean flood probability
        variance:   (H, W) predictive variance
        trust_mask: (H, W) bool
        label:      (H, W) int ground truth (0/1/-1)
        chip_id:    string identifier for title
        out_path:   file path to save PNG
    """
    # IEEE double-column width: 7.16 in; 4 panels → 7.2 × 3.6 in
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8))

    var_max = float(variance.max()) or 0.1
    panels = [
        # (data,                       title,                  cmap,        vmin,    vmax,    cbar_label)
        (mean_prob,                 "Flood probability",    "RdYlBu_r",  0.0,     1.0,     "P(flood)"),
        (variance,                  "Predictive variance",  "YlOrRd",    0.0,     var_max, "Var"),
        (trust_mask.astype(float),  "Trust mask",           "Greens",    0.0,     1.0,     ""),
        ((label > 0).astype(float), "Ground truth",         "Blues",     0.0,     1.0,     ""),
    ]

    for ax, (data, title, cmap, vmin, vmax, cbar_label) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, pad=3)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)
        cb.ax.tick_params(labelsize=7)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=7)

    fig.suptitle(f"Chip: {chip_id}", y=1.01)
    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 2.  Reliability diagram
# ─────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    bin_accs:  np.ndarray,
    bin_confs: np.ndarray,
    ece:       float,
    out_path:  str,
    title:     str = "Reliability Diagram",
) -> None:
    """
    Reliability diagram (calibration curve).
    Perfect calibration = diagonal line.

    Args:
        bin_accs:  (n_bins,) mean accuracy per confidence bin
        bin_confs: (n_bins,) mean confidence per bin
        ece:       Expected Calibration Error (scalar)
        out_path:  save path
        title:     plot title
    """
    n_bins      = len(bin_accs)
    bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
    bin_width   = 1.0 / n_bins

    # IEEE single-column: 3.5 in square
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Gap fill: shade over-/under-confidence regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.08,
                    color="#d32f2f", label="Over-confident")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.08,
                    color="#1565c0", label="Under-confident")

    ax.bar(bin_centers, bin_accs, width=bin_width * 0.9, alpha=0.55,
           color="#1976D2", label="Model", edgecolor="white", linewidth=0.5)
    ax.plot(bin_centers, bin_accs, "o-", color="#0D47A1",
            linewidth=1.5, markersize=4, zorder=5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2,
            label="Perfect calibration", zorder=4)

    # ECE annotation — placed inside axes, top-left
    ax.text(0.04, 0.93, f"ECE = {ece:.4f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))

    ax.set_xlabel("Confidence (predicted probability)")
    ax.set_ylabel("Fraction of flood pixels")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 3.  Coverage-accuracy curve
# ─────────────────────────────────────────────────────────────

def plot_coverage_accuracy(
    results:       list,
    out_path:      str,
    n_thresh:      int = 50,
    variant_label: str = "Variant D (HAND gate + MC Dropout)",
) -> None:
    """
    Coverage-accuracy curve.

    Sweeps variance thresholds. At each threshold:
      coverage = fraction of valid pixels with variance <= threshold
      accuracy = IoU of flood predictions inside the trust mask

    Args:
        results:       list of per-chip dicts (mean_prob, variance, label)
        out_path:      save path
        n_thresh:      number of threshold steps
        variant_label: legend label — pass e.g. "Variant D (HAND gate + MC Dropout)"
                       so figures are correctly labelled when called for each variant
    """
    all_probs  = []
    all_vars   = []
    all_labels = []
    for r in results:
        valid = r["label"] != -1
        all_probs.append(r["mean_prob"][valid].flatten())
        all_vars.append(r["variance"][valid].flatten())
        all_labels.append(r["label"][valid].flatten())

    probs     = np.concatenate(all_probs)
    variances = np.concatenate(all_vars)
    labels    = np.concatenate(all_labels).astype(np.float32)

    # Guard against degenerate case (Variants A/B/C: dropout_rate=0 → variance=0)
    # np.linspace(0, 0, n) gives all-zero thresholds but trusted=all True for
    # tau=0 when variances are 0 — it degenerates gracefully to a single point
    # at (coverage=1.0, IoU=constant). The `or 1.0` ensures thresholds sweep
    # [0, 1] even when max_var=0 so the plot always has meaningful axes.
    max_var    = float(variances.max()) or 1.0
    thresholds = np.linspace(0, max_var, n_thresh + 1)[1:]

    coverages  = []
    accuracies = []

    for thr in thresholds:
        trusted = variances <= thr
        if trusted.sum() == 0:
            continue
        cov   = float(trusted.mean())
        preds = (probs[trusted] > 0.5).astype(np.float32)
        labs  = labels[trusted]
        inter = ((preds == 1) & (labs == 1)).sum()
        union = ((preds == 1) | (labs == 1)).sum()
        iou   = float(inter) / max(float(union), 1)
        coverages.append(cov)
        accuracies.append(iou)

    # IEEE single-column width
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(coverages, accuracies, "o-", color="#1565C0",
            linewidth=1.5, markersize=3.5,
            label=variant_label)
    ax.axvline(x=0.7, color="#9E9E9E", linestyle="--",
               linewidth=1.0, alpha=0.8, label="Coverage = 0.70")

    ax.set_xlabel("Coverage (fraction of trusted pixels)")
    ax.set_ylabel("IoU on trusted pixels")
    ax.set_title("Coverage–Accuracy Trade-off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(fontsize=8)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 4.  IoU bar chart (per event)
# ─────────────────────────────────────────────────────────────

def plot_iou_bar_chart(
    per_event_metrics: dict,
    out_path:          str,
    metric:            str = "iou",
    title:             str = "Flood IoU per Event",
) -> None:
    """
    Horizontal bar chart of IoU (or other metric) per flood event.

    Args:
        per_event_metrics: {event_name: {iou: float, ...}, ...}
        out_path:          save path
        metric:            key to plot from each event dict
        title:             plot title
    """
    events  = sorted(per_event_metrics.keys())
    values  = [per_event_metrics[e].get(metric, 0.0) for e in events]

    # Colour: green ≥ 0.5 (good), amber 0.3–0.5, red < 0.3
    colours = []
    for v in values:
        if v >= 0.50:
            colours.append("#2E7D32")   # dark green
        elif v >= 0.30:
            colours.append("#E65100")   # dark orange
        else:
            colours.append("#B71C1C")   # dark red

    # Single-column width; height scales with number of events
    fig, ax = plt.subplots(figsize=(3.5, max(2.8, len(events) * 0.32 + 0.8)))
    bars = ax.barh(events, values, color=colours, edgecolor="white",
                   linewidth=0.4, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(min(val + 0.01, 1.05), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=7.5,
                color="#212121")

    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.set_xlim(0, 1.12)
    ax.axvline(x=0.5, color="#757575", linestyle="--",
               linewidth=0.8, alpha=0.7, label=f"{metric.upper()} = 0.50")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 5.  Risk-Coverage curve
# ─────────────────────────────────────────────────────────────

def plot_risk_coverage_curve(
    rc_results:       dict | list[dict],
    out_path:         str,
    variant_labels:   list[str] | None = None,
) -> None:
    """
    Risk-Coverage curve: sweeps uncertainty threshold τ and at each τ plots
      X: coverage  = fraction of pixels with variance ≤ τ
      Y: risk      = 1 − IoU computed only on trusted pixels

    A better model hugs the bottom-right corner (high coverage, low risk).
    AURC (area under the curve) is annotated; lower AURC is better.

    Args:
        rc_results:     single dict OR list of dicts from compute_risk_coverage_curve()
                        Each dict must have keys: coverage, risk, aurc, and optionally label.
        out_path:       save path
        variant_labels: optional list of legend labels when comparing multiple variants
    """
    if isinstance(rc_results, dict):
        rc_results = [rc_results]

    # Colour-blind-safe palette (Okabe–Ito)
    colours = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    default_labels = ["Variant A", "Variant B", "Variant C", "Variant D"]

    # IEEE single-column width
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for idx, rc in enumerate(rc_results):
        coverage = np.asarray(rc["coverage"])
        risk     = np.asarray(rc["risk"])
        aurc     = rc["aurc"]
        lbl      = (variant_labels or default_labels)[idx] \
                   if (variant_labels or default_labels) else f"Variant {idx}"
        colour   = colours[idx % len(colours)]
        marker   = markers[idx % len(markers)]

        ax.plot(coverage, risk, linestyle="-", marker=marker,
                color=colour, linewidth=1.5, markersize=3,
                markevery=max(1, len(coverage) // 10),
                label=f"{lbl}  (AURC = {aurc:.4f})")

    ax.set_xlabel("Coverage (fraction of pixels retained)")
    ax.set_ylabel("Risk  (1 − IoU on retained pixels)")
    ax.set_title("Risk–Coverage Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Standard convention: x-axis decreasing (most selective → least selective)
    ax.invert_xaxis()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(fontsize=7.5, loc="upper left")

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 6.  Ablation comparison figure
# ─────────────────────────────────────────────────────────────

def plot_ablation_table(
    ablation_results: dict,
    out_path:         str,
) -> None:
    """
    Side-by-side bar chart comparing all 4 ablation variants.

    Args:
        ablation_results: {
            "A": {"iou": 0.45, "ece": 0.08, "brier": 0.12},
            "B": {...},
            "C": {...},
            "D": {...},
        }
        out_path: save path
    """
    variants = ["A", "B", "C", "D"]
    # Short labels to fit single-column width; use \n for two-line ticks
    tick_labels = [
        "A\n(SAR only)",
        "B\n(+HAND band)",
        "C\n(+HAND gate)",
        "D\n(+HAND gate\n+MC Dropout)",
    ]
    metrics = ["iou", "f1", "ece", "brier"]
    titles  = ["IoU ↑", "F1 ↑", "ECE ↓", "Brier Score ↓"]

    x = np.arange(len(variants))

    # IEEE double-column: 4 panels, 7.16 in wide × 2.8 in tall
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.8))

    # Okabe–Ito colour-blind-safe palette
    base_colour = "#90CAF9"     # light blue for A/B/C
    highlight   = "#1565C0"     # dark blue for D (our full model)

    for ax, metric, mtitle in zip(axes, metrics, titles):
        vals    = [ablation_results.get(v, {}).get(metric, 0.0) for v in variants]
        colours = [highlight if v == "D" else base_colour for v in variants]
        bars    = ax.bar(x, vals, color=colours, edgecolor="white",
                         linewidth=0.4, width=0.65)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals or [0]) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_title(mtitle)
        top = max(vals or [0.1]) * 1.30 + 0.02
        ax.set_ylim(0, max(top, 0.12))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle("Ablation Study — Bolivia OOD Test Set")
    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 7.  HAND gate attention map visualisation
# ─────────────────────────────────────────────────────────────

def plot_hand_gate_maps(
    sar_vv:    np.ndarray,
    hand_m:    np.ndarray,
    gate_maps: list,
    label:     np.ndarray,
    mean_prob: np.ndarray,
    chip_id:   str,
    out_path:  str,
) -> None:
    """
    Visualise the HAND attention gate maps alongside SAR and ground truth.

    Produces a 3-row publication figure:
      Row 1: SAR VV (post)  |  HAND in metres  |  Flood probability  |  GT label
      Row 2: Gate α — level 0 (finest, H/2) ... level 3 (coarsest, H/32)
      Row 3: (title labels)

    Physical interpretation of α values:
      α → 1.0  (bright)  = near river / low HAND  → flood signal passes through
      α → 0.0  (dark)    = high HAND / hillside   → feature suppressed

    Args:
        sar_vv:    (H, W) SAR VV post-event (raw dB)
        hand_m:    (H, W) HAND in metres (denormalised)
        gate_maps: list of 4 arrays, each (H, W) with α values ∈ [0, 1]
                   ordered finest → coarsest  [alpha0, alpha1, alpha2, alpha3]
        label:     (H, W) ground truth (0/1/-1 ignore)
        mean_prob: (H, W) flood probability from MC inference
        chip_id:   string identifier for suptitle
        out_path:  save path
    """
    # ── stretch SAR for display ───────────────────────────────
    def _stretch(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.nanpercentile(arr, [2, 98])
        if abs(hi - lo) < 1e-8:
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    # ── figure layout: 2 rows × 4 cols ───────────────────────
    # IEEE double-column: 7.16 in wide
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))

    gate_titles = [
        r"Gate $\alpha_0$  (H/2)",
        r"Gate $\alpha_1$  (H/4)",
        r"Gate $\alpha_2$  (H/8)",
        r"Gate $\alpha_3$  (H/32)",
    ]

    # ── Row 0: context panels ────────────────────────────────
    row0_panels = [
        (_stretch(sar_vv),              "SAR VV (post)",   "gray",     None,  None),
        (hand_m,                        "HAND (metres)",   "terrain",  0,     None),
        (mean_prob,                     "P(flood)",        "RdYlBu_r", 0.0,   1.0),
        ((label > 0).astype(np.float32),"Ground truth",   "Blues",    0.0,   1.0),
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes[0], row0_panels):
        kw = {"cmap": cmap, "interpolation": "nearest"}
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax is not None:
            kw["vmax"] = vmax
        im = ax.imshow(data, **kw)
        ax.set_title(title, pad=3)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.82)
        cb.ax.tick_params(labelsize=6)

    # ── Row 1: gate α maps ───────────────────────────────────
    for ax, alpha, title in zip(axes[1], gate_maps, gate_titles):
        im = ax.imshow(alpha, cmap="RdYlGn",
                       vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax.set_title(title, pad=3)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.82)
        cb.ax.tick_params(labelsize=6)
        cb.set_ticks([0.0, 0.5, 1.0])

    fig.suptitle(
        f"HAND Gate Attention Maps — Chip: {chip_id}\n"
        r"Green $\alpha\!\approx\!1$ (flood-prone) · Red $\alpha\!\approx\!0$ (suppressed)",
        y=1.01, fontsize=10,
    )
    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
