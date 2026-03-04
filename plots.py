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
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    var_max = float(variance.max()) or 0.1
    panels = [
        (mean_prob,                 "Flood probability",   "RdYlBu_r", 0, 1),
        (variance,                  "Predictive variance", "hot_r",     0, var_max),
        (trust_mask.astype(float),  "Trust mask",          "Greens",    0, 1),
        ((label > 0).astype(float), "Ground truth",        "Blues",     0, 1),
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax.bar(bin_centers, bin_accs, width=1 / n_bins, alpha=0.6,
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


# ─────────────────────────────────────────────────────────────
# 3.  Coverage-accuracy curve
# ─────────────────────────────────────────────────────────────

def plot_coverage_accuracy(
    results:  list,
    out_path: str,
    n_thresh: int = 50,
) -> None:
    """
    Coverage-accuracy curve.

    Sweeps variance thresholds. At each threshold:
      coverage = fraction of valid pixels with variance <= threshold
      accuracy = IoU of flood predictions inside the trust mask

    Args:
        results:  list of per-chip dicts (mean_prob, variance, label)
        out_path: save path
        n_thresh: number of threshold steps
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

    max_var    = float(variances.max())
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

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, accuracies, "o-", color="steelblue", linewidth=2,
            markersize=4, label="Model D (HAND + MC Dropout)")
    ax.set_xlabel("Coverage (fraction of trusted pixels)", fontsize=12)
    ax.set_ylabel("IoU (flood class, trusted pixels only)", fontsize=12)
    ax.set_title("Coverage–Accuracy Tradeoff", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axvline(x=0.7, color="gray", linestyle="--", alpha=0.5,
               label="Coverage target (0.7)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    colours = ["#2196F3" if v >= 0.5 else "#FF9800" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(events) * 0.5 + 1)))
    bars = ax.barh(events, values, color=colours, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(min(val + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="IoU = 0.5")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 5.  Ablation comparison figure
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
    labels   = ["A\n(SAR only)", "B\n(+HAND band)", "C\n(+HAND gate)", "D\n(+MC Dropout)"]
    metrics  = ["iou", "ece", "brier"]
    titles   = ["IoU ↑", "ECE ↓", "Brier ↓"]

    x = np.arange(len(variants))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, title in zip(axes, metrics, titles):
        vals    = [ablation_results.get(v, {}).get(metric, 0.0) for v in variants]
        colours = ["#4CAF50" if v == "D" else "#90CAF9" for v in variants]
        bars    = ax.bar(x, vals, color=colours, edgecolor="white", width=0.6)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(vals or [0.1]) * 1.25 + 0.05)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Ablation Study — All Variants", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
