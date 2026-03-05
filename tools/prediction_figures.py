#!/usr/bin/env python3
"""
Export qualitative prediction figures directly from TensorBoard event files.

This avoids importing model code entirely and is the most robust way to generate
paper-ready prediction visuals on HPC (DKUCC).

It reads the TensorBoard image tag (default: val/sar_pred_label) and saves:
- A grid figure with selected training steps
- Optionally, individual PNGs for each step

Run:
  cd ~/terrainflood
  python tools/prediction_figures.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ----------------------------
# CONFIG (edit if you want)
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]  # .../terrainflood

LOGDIR = ROOT / "checkpoints" / "variant_B" / "runs"
OUT_DIR = ROOT / "results" / "paper_figures"

TB_IMAGE_TAG = "val/sar_pred_label"

# How many steps to show in the grid figure:
N_GRID = 2  # you can set to 3 or 4 if you want

# Save each step as an individual PNG too?
SAVE_INDIVIDUAL = True


# ----------------------------
# Helpers
# ----------------------------
def _load_tb_images(logdir: Path, tag: str) -> List[Tuple[int, np.ndarray]]:
    """
    Returns a list of (step, image_array) sorted by step.
    image_array is uint8 RGB or RGBA.
    """
    ea = EventAccumulator(str(logdir))
    ea.Reload()

    tags = ea.Tags()
    if "images" not in tags or tag not in tags["images"]:
        raise RuntimeError(
            f"TensorBoard tag '{tag}' not found.\n"
            f"Available image tags: {tags.get('images', [])}\n"
            f"Logdir: {logdir}"
        )

    img_events = ea.Images(tag)
    out: List[Tuple[int, np.ndarray]] = []

    for ev in img_events:
        step = int(ev.step)
        # ev.encoded_image_string is a PNG/JPEG byte string
        pil = Image.open(io.BytesIO(ev.encoded_image_string))
        arr = np.array(pil)
        out.append((step, arr))

    out.sort(key=lambda x: x[0])
    return out


def _save_grid(images: List[Tuple[int, np.ndarray]], out_path: Path, title: str) -> None:
    """
    Save a grid of selected steps (first/last or evenly spaced).
    """
    if len(images) == 0:
        raise RuntimeError("No images found to plot.")

    # Choose steps (evenly spaced)
    if N_GRID <= 1:
        idxs = [len(images) - 1]
    else:
        idxs = np.linspace(0, len(images) - 1, N_GRID).round().astype(int).tolist()

    chosen = [images[i] for i in idxs]

    # Make a 1-row grid
    fig_w = 7.0 * len(chosen)
    fig_h = 5.5
    fig, axes = plt.subplots(1, len(chosen), figsize=(fig_w, fig_h), constrained_layout=True)
    if len(chosen) == 1:
        axes = [axes]

    for ax, (step, arr) in zip(axes, chosen):
        ax.imshow(arr)
        ax.set_title(f"Step {step}", fontsize=18)
        ax.axis("off")

    fig.suptitle(title, fontsize=22)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _save_individual(images: List[Tuple[int, np.ndarray]], out_dir: Path, prefix: str) -> None:
    """
    Save each step's image as its own PNG.
    """
    for step, arr in images:
        out_path = out_dir / f"{prefix}_step{step:06d}.png"
        pil = Image.fromarray(arr)
        pil.save(out_path)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LOGDIR.exists():
        raise FileNotFoundError(f"LOGDIR not found: {LOGDIR}")

    images = _load_tb_images(LOGDIR, TB_IMAGE_TAG)

    # Save grid (paper-ready)
    grid_path = OUT_DIR / "fig7_predictions_from_tb.png"
    title = "TerrainFlood-UQ | Variant B: Qualitative predictions (TensorBoard tag: val/sar_pred_label)"
    _save_grid(images, grid_path, title)

    # Save all individual frames (optional)
    if SAVE_INDIVIDUAL:
        indiv_dir = OUT_DIR / "pred_frames_variantB"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        _save_individual(images, indiv_dir, prefix="variantB_pred")

    print(f"[OK] Loaded {len(images)} images from TB tag '{TB_IMAGE_TAG}'")
    print(f"[OK] Saved grid figure: {grid_path}")
    if SAVE_INDIVIDUAL:
        print(f"[OK] Saved individual frames to: {OUT_DIR / 'pred_frames_variantB'}")


if __name__ == "__main__":
    main()