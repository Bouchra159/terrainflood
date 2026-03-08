import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

OUT_DIR = "results/paper_remote_sensing_maps"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Utility: load image safely
# ---------------------------------------------------

def load_img(path):
    try:
        return np.array(Image.open(path))
    except:
        return None

# ---------------------------------------------------
# Find example images automatically
# ---------------------------------------------------

pred_imgs = glob("results/eval_D/figures/*.png")
uncert_imgs = glob("results/uncertainty_*/*.png")

pred = load_img(pred_imgs[0]) if pred_imgs else None
uncert = load_img(uncert_imgs[0]) if uncert_imgs else None

# ---------------------------------------------------
# Figure 1: Flood Prediction Visualization
# ---------------------------------------------------

if pred is not None:

    plt.figure(figsize=(6,6))
    plt.imshow(pred)
    plt.title("Flood Segmentation Prediction (Variant D)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig05_flood_prediction_map.png", dpi=300)
    plt.close()

# ---------------------------------------------------
# Figure 2: Uncertainty Visualization
# ---------------------------------------------------

if uncert is not None:

    plt.figure(figsize=(6,6))
    plt.imshow(uncert, cmap="inferno")
    plt.title("Predictive Uncertainty Map")
    plt.axis("off")

    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig06_uncertainty_map.png", dpi=300)
    plt.close()

print("Remote sensing maps saved to:", OUT_DIR)
