import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results")
OUT = ROOT / "paper_figures_from_json"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------- load ----------
ablation = load_json(ROOT / "ablation" / "ablation_metrics.json")
bootstrap = load_json(ROOT / "bootstrap_ci" / "ablation_stats.json")
thresholds = load_json(ROOT / "threshold_sweep" / "optimal_thresholds.json")
alpha = load_json(ROOT / "alpha_vs_hand" / "alpha_vs_hand.json")

# ---------- Figure 1: Ablation IoU bar ----------
preferred_order = ["A", "B", "C", "D", "D_plus", "E"]
variants = [v for v in preferred_order if v in ablation]

ious = [ablation[v]["iou"] for v in variants]
f1s = [ablation[v]["f1"] for v in variants]

x = np.arange(len(variants))
plt.figure(figsize=(8, 4.8))
bars = plt.bar(x, ious)
plt.xticks(x, variants)
plt.ylabel("IoU")
plt.ylim(0, max(0.8, max(ious) + 0.05))
plt.title("Bolivia OOD Ablation Performance")
for i, v in enumerate(ious):
    plt.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
savefig(OUT / "fig1_ablation_iou.png")

# ---------- Figure 2: Bootstrap CI ----------
ci_variants = [v for v in preferred_order if v in bootstrap["variants"]]
obs = [bootstrap["variants"][v]["ci"]["iou_observed"] for v in ci_variants]
low = [bootstrap["variants"][v]["ci"]["ci_lower"] for v in ci_variants]
high = [bootstrap["variants"][v]["ci"]["ci_upper"] for v in ci_variants]
yerr = np.array([
    [o - l for o, l in zip(obs, low)],
    [h - o for o, h in zip(obs, high)],
])

x = np.arange(len(ci_variants))
plt.figure(figsize=(8, 4.8))
plt.bar(x, obs)
plt.errorbar(x, obs, yerr=yerr, fmt="none", capsize=5)
plt.xticks(x, ci_variants)
plt.ylabel("IoU")
plt.ylim(0, max(0.85, max(high) + 0.05))
plt.title("Bootstrap 95% Confidence Intervals")
for i, v in enumerate(obs):
    plt.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
savefig(OUT / "fig2_bootstrap_ci.png")

# ---------- Figure 3: Threshold sweep summary ----------
thr_variants = [v for v in preferred_order if v in thresholds]
tau = [thresholds[v]["tau_star"] for v in thr_variants]
iou_tau = [thresholds[v]["iou_at_tau"] for v in thr_variants]
iou_05 = [thresholds[v]["iou_at_0.5"] for v in thr_variants]

x = np.arange(len(thr_variants))
w = 0.35
plt.figure(figsize=(9, 4.8))
plt.bar(x - w/2, iou_05, width=w, label="IoU @ 0.5")
plt.bar(x + w/2, iou_tau, width=w, label="IoU @ tau*")
plt.xticks(x, [f"{v}\nτ*={t:.2f}" for v, t in zip(thr_variants, tau)])
plt.ylabel("IoU")
plt.ylim(0, max(0.85, max(iou_tau) + 0.05))
plt.title("Threshold Optimization Summary")
plt.legend()
savefig(OUT / "fig3_threshold_summary.png")

# ---------- Figure 4: Alpha vs HAND ----------
alpha_variants = [v for v in ["C", "D", "D_plus", "E"] if v in alpha]
plt.figure(figsize=(8, 5))
for v in alpha_variants:
    bins = alpha[v]["bin_centres_m"]
    mean_alpha = alpha[v]["mean_alpha"]
    xs, ys = [], []
    for bx, by in zip(bins, mean_alpha):
        if by is not None:
            xs.append(bx)
            ys.append(by)
    if xs:
        plt.plot(xs, ys, marker="o", label=f"{v} (r={alpha[v]['pearson_r_theory']:+.3f})")
plt.xlabel("HAND elevation bin center (m)")
plt.ylabel("Mean gate alpha")
plt.title("Gate Response vs HAND Elevation")
plt.legend()
savefig(OUT / "fig4_alpha_vs_hand.png")

# ---------- Table-friendly summary markdown ----------
summary_md = OUT / "paper_results_summary.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# TerrainFlood-UQ Results Summary\n\n")
    f.write("## Bolivia OOD Ablation\n\n")
    f.write("| Variant | IoU | F1 | Precision | Recall | ECE | Mean Variance |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for v in variants:
        m = ablation[v]
        f.write(
            f"| {v} | {m['iou']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['ece']:.4f} | {m['mean_variance']:.4f} |\n"
        )

    f.write("\n## Bootstrap 95% CI\n\n")
    f.write("| Variant | IoU | CI Lower | CI Upper |\n")
    f.write("|---|---:|---:|---:|\n")
    for v in ci_variants:
        c = bootstrap["variants"][v]["ci"]
        f.write(f"| {v} | {c['iou_observed']:.4f} | {c['ci_lower']:.4f} | {c['ci_upper']:.4f} |\n")

    f.write("\n## Threshold Optimization\n\n")
    f.write("| Variant | tau* | IoU@tau* | IoU@0.5 |\n")
    f.write("|---|---:|---:|---:|\n")
    for v in thr_variants:
        t = thresholds[v]
        f.write(f"| {v} | {t['tau_star']:.2f} | {t['iou_at_tau']:.4f} | {t['iou_at_0.5']:.4f} |\n")

    f.write("\n## Alpha vs HAND\n\n")
    f.write("| Variant | Pearson r vs theory |\n")
    f.write("|---|---:|\n")
    for v in alpha_variants:
        f.write(f"| {v} | {alpha[v]['pearson_r_theory']:+.4f} |\n")

    f.write("\n## Short interpretation\n\n")
    f.write("- Variant D is the best overall Bolivia OOD model.\n")
    f.write("- Variant C is the strongest non-dropout gated model.\n")
    f.write("- D_plus increases uncertainty magnitude but trails D on IoU.\n")
    f.write("- Variant E underperforms strongly in the current temporal-differencing setup.\n")
    f.write("- Threshold tuning does not change the top-ranked model.\n")

print(f"Saved figures and summary to: {OUT}")
