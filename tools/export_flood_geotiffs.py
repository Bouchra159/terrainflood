#!/usr/bin/env python3
"""
Georeferenced Flood Map Export
File: tools/export_flood_geotiffs.py

Runs Variant D inference on the requested split and writes each chip's
flood probability map as a georeferenced GeoTIFF — ready to open in
QGIS, ArcGIS Pro, or Google Earth Engine.

Output GeoTIFF properties:
  - Single band: float32, values ∈ [0, 1] (flood probability)
  - Same CRS and pixel resolution as the input S1Hand.tif
  - nodata = -9999 for ignore-mask pixels (label == -1)
  - Filename: {event}_{chip_id}_flood_prob.tif

These outputs can be:
  1. Opened directly in QGIS / ArcGIS Pro
  2. Uploaded to GEE as Image assets for interactive visualisation
  3. Used to compute flood area in km² from pixel area + CRS info
  4. Overlaid with Copernicus EMS reference maps for benchmark comparison

Usage:
  python tools/export_flood_geotiffs.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/flood_geotiffs \\
      --split      test \\
      --T          20

  # Val set:
  python tools/export_flood_geotiffs.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/flood_geotiffs_val \\
      --split      val \\
      --T          20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


# ─────────────────────────────────────────────────────────────
# Numeric-prefix imports
# ─────────────────────────────────────────────────────────────

_root = Path(__file__).parent.parent


def _import_module(alias: str, file_path: str):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "model_gtiff" not in sys.modules:
    _import_module("model_gtiff",   str(_root / "03_model.py"))
if "dataset_gtiff" not in sys.modules:
    _import_module("dataset_gtiff", str(_root / "02_dataset.py"))

from model_gtiff   import build_model      # noqa: E402
from dataset_gtiff import get_dataloaders  # noqa: E402

NODATA = -9999.0


# ─────────────────────────────────────────────────────────────
# Spatial metadata from original S1 chip
# ─────────────────────────────────────────────────────────────

def _get_spatial_meta(s1_path: Path, target_height: int, target_width: int) -> dict:
    """
    Read CRS and transform from the original S1Hand.tif.
    Rescales transform if the model resized the chip.

    Args:
        s1_path:       path to the original *_S1Hand.tif
        target_height: model output height (pixels)
        target_width:  model output width  (pixels)

    Returns:
        rasterio-compatible profile dict
    """
    with rasterio.open(s1_path) as src:
        crs       = src.crs
        transform = src.transform
        src_h     = src.height
        src_w     = src.width

    # If the model output has a different resolution, rescale the transform
    # so the GeoTIFF covers the same geographic extent.
    if (src_h, src_w) != (target_height, target_width):
        bounds = rasterio.transform.array_bounds(src_h, src_w, transform)
        transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            target_width, target_height,
        )

    return {
        "driver":    "GTiff",
        "dtype":     "float32",
        "nodata":    NODATA,
        "width":     target_width,
        "height":    target_height,
        "count":     1,
        "crs":       crs,
        "transform": transform,
        "compress":  "lzw",
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    }


# ─────────────────────────────────────────────────────────────
# Export loop
# ─────────────────────────────────────────────────────────────

def export_geotiffs(
    ckpt_path:   str,
    data_root:   str,
    output_dir:  str,
    device:      torch.device,
    split:       str  = "test",
    T:           int  = 20,
    batch_size:  int  = 1,
    num_workers: int  = 2,
) -> None:
    """
    Run MC Dropout inference and write one GeoTIFF per chip.

    Args:
        ckpt_path:    path to best.pt
        data_root:    Sen1Floods11 root
        output_dir:   directory to write *_flood_prob.tif files
        device:       torch device
        split:        "test" or "val"
        T:            MC Dropout passes (use 1 for deterministic export)
        batch_size:   keep at 1 to preserve spatial metadata per chip
        num_workers:  dataloader workers
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root_path = Path(data_root)
    s1_dir = data_root_path / "flood_events" / "HandLabeled" / "S1Hand"

    # ── Load checkpoint ──────────────────────────────────────
    ckpt    = torch.load(ckpt_path, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    if variant == "D":
        model.enable_dropout()

    print(f"Loaded Variant {variant}  epoch={ckpt['epoch']}  "
          f"best_iou={ckpt.get('best_iou', 0.0):.4f}")
    print(f"Split: {split}  T={T}  Output: {out_dir}\n")

    # ── Dataloader ───────────────────────────────────────────
    _, val_loader, test_loader = get_dataloaders(
        data_root   = data_root,
        batch_size  = batch_size,
        num_workers = num_workers,
        patch_size  = None,
        pin_memory  = False,
    )
    loader = test_loader if split == "test" else val_loader

    # ── Inference loop ───────────────────────────────────────
    summary: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device)     # (1, 6, H, W)
            labels  = batch["label"].numpy()        # (1, H, W)
            chip_id = batch["chip_id"][0]
            event   = batch["event"][0]

            H, W = images.shape[2], images.shape[3]

            # MC Dropout: average T passes
            probs_mc = []
            for _ in range(T):
                logits = model(images)          # (1, 1, H, W)
                prob   = torch.sigmoid(logits.squeeze(1)).cpu().numpy()  # (1, H, W)
                probs_mc.append(prob)

            mean_prob = np.mean(probs_mc, axis=0)[0]   # (H, W)
            label_2d  = labels[0]                       # (H, W)

            # Mask nodata pixels
            flood_map = mean_prob.astype(np.float32)
            flood_map[label_2d == -1] = NODATA

            # Compute flood area (valid pixels only, prob > 0.5)
            flood_pixels = int(((mean_prob > 0.5) & (label_2d != -1)).sum())

            # Locate original S1Hand.tif for spatial metadata
            s1_path = s1_dir / f"{chip_id}_S1Hand.tif"
            if not s1_path.exists():
                print(f"  [WARN] S1 not found for {chip_id} — writing without CRS")
                meta = {
                    "driver": "GTiff", "dtype": "float32", "nodata": NODATA,
                    "width": W, "height": H, "count": 1, "crs": None,
                    "transform": rasterio.transform.from_bounds(0, 0, W, H, W, H),
                }
            else:
                meta = _get_spatial_meta(s1_path, H, W)

            # Write GeoTIFF
            out_fname = f"{event}_{chip_id}_flood_prob.tif"
            out_path  = out_dir / out_fname
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(flood_map[np.newaxis, :, :])   # (1, H, W)
                dst.update_tags(
                    CHIP_ID       = chip_id,
                    EVENT         = event,
                    MODEL_VARIANT = variant,
                    MC_PASSES     = str(T),
                    DESCRIPTION   = "Flood probability [0,1]; nodata=-9999",
                )

            summary.append({
                "chip_id":       chip_id,
                "event":         event,
                "file":          out_fname,
                "flood_pixels":  flood_pixels,
                "has_crs":       s1_path.exists(),
            })

            print(f"  {chip_id:<35}  flood_px={flood_pixels:6d}  → {out_fname}")

    # ── Summary JSON ─────────────────────────────────────────
    summary_path = out_dir / "geotiff_summary.json"
    summary_path.write_text(json.dumps({
        "variant":    variant,
        "checkpoint": ckpt_path,
        "split":      split,
        "T":          T,
        "n_chips":    len(summary),
        "chips":      summary,
        "qgis_tip":   "Open any *_flood_prob.tif in QGIS. Use Singleband Pseudocolor "
                      "with range 0–1 and a blue/white colormap for best visualisation.",
    }, indent=2))

    print(f"\nExported {len(summary)} GeoTIFFs → {out_dir}")
    print(f"Summary → {summary_path}")
    print("\nTo open in QGIS:")
    print(f"  Layer → Add Layer → Add Raster Layer → {out_dir}/*.tif")
    print("  Style: Singleband pseudocolor, range 0–1, Blues colormap")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export flood probability maps as georeferenced GeoTIFFs"
    )
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to best.pt")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/flood_geotiffs")
    p.add_argument("--split",        type=str, default="test",
                   choices=["test", "val"])
    p.add_argument("--T",            type=int, default=20,
                   help="MC Dropout passes (use 1 for deterministic export)")
    p.add_argument("--num_workers",  type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    export_geotiffs(
        ckpt_path   = args.checkpoint,
        data_root   = args.data_root,
        output_dir  = args.output_dir,
        device      = device,
        split       = args.split,
        T           = args.T,
        num_workers = args.num_workers,
    )
