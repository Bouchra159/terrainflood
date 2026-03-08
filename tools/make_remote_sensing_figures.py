import json
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "results/paper_remote_sensing_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load experiment outputs
# -------------------------

with open("results/ablation/ablation_metrics.json") as f:
    ablation = json.load(f)

with open("results/bootstrap_ci/ablation_stats.json") as f:
    bootstrap = json.load(f)

with open("results/threshold_sweep/optimal_thresholds.json") as f:
    thresholds = json.load(f)

with open("results/alpha_vs_hand/alpha_vs_hand.json") as f:
    alpha = json.load(f)

# -------------------------
# Figure 1 — Ablation IoU
# -------------------------

variants = ["A","B","C","D","D_plus","E"]
ious = [ablation[v]["iou"] for v in variants if v in ablation]

plt.figure(figsize=(8,5))
plt.bar(variants, ious)
plt.ylabel("Intersection over Union (IoU)")
plt.title("Flood Segmentation Ablation Study (Bolivia OOD)")
plt.ylim(0,0.8)

for i,v in enumerate(ious):
    plt.text(i, v+0.01, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig01_ablation_iou.png", dpi=300)
plt.close()

# -------------------------
# Figure 2 — Bootstrap CI
# -------------------------

variants_ci = list(bootstrap["variants"].keys())

obs = [bootstrap["variants"][v]["ci"]["iou_observed"] for v in variants_ci]
low = [bootstrap["variants"][v]["ci"]["ci_lower"] for v in variants_ci]
high = [bootstrap["variants"][v]["ci"]["ci_upper"] for v in variants_ci]

errors = [np.array(obs)-np.array(low), np.array(high)-np.array(obs)]

plt.figure(figsize=(8,5))
plt.bar(variants_ci, obs)
plt.errorbar(variants_ci, obs, yerr=errors, fmt='none', capsize=5)

plt.ylabel("IoU")
plt.title("Bootstrap 95% Confidence Intervals")
plt.ylim(0,0.85)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig02_bootstrap_ci.png", dpi=300)
plt.close()

# -------------------------
# Figure 3 — Threshold sweep
# -------------------------

variants_th = list(thresholds.keys())
tau = [thresholds[v]["tau_star"] for v in variants_th]
iou_tau = [thresholds[v]["iou_at_tau"] for v in variants_th]
iou05 = [thresholds[v]["iou_at_0.5"] for v in variants_th]

x = np.arange(len(variants_th))

plt.figure(figsize=(9,5))
plt.bar(x-0.2, iou05, width=0.4, label="IoU @ 0.5")
plt.bar(x+0.2, iou_tau, width=0.4, label="IoU @ τ*")

plt.xticks(x, variants_th)
plt.ylabel("IoU")
plt.title("Threshold Optimization Across Model Variants")
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig03_threshold_sweep.png", dpi=300)
plt.close()

# -------------------------
# Figure 4 — Alpha vs HAND
# -------------------------

plt.figure(figsize=(8,5))

for v in alpha:
    bins = alpha[v]["bin_centres_m"]
    mean = alpha[v]["mean_alpha"]

    xs=[]
    ys=[]
    for b,m in zip(bins,mean):
        if m is not None:
            xs.append(b)
            ys.append(m)

    if len(xs)>0:
        r = alpha[v]["pearson_r_theory"]
        plt.plot(xs,ys,marker="o",label=f"{v} (r={r:+.2f})")

plt.xlabel("HAND Elevation (m)")
plt.ylabel("Gate Activation α")
plt.title("Relationship Between HAND Terrain Elevation and Learned Gating")
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig04_alpha_vs_hand.png", dpi=300)
plt.close()

print("Figures saved to:", OUT_DIR)
