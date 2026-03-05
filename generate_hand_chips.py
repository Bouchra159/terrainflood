#!/usr/bin/env python3
"""
generate_hand_chips.py
======================
Generate per-chip HAND (Height Above Nearest Drainage) GeoTIFFs
for ALL Sen1Floods11 HandLabeled chips.

What it does:
  1. For each S1 chip, reads its geographic footprint (bounds + CRS)
  2. Downloads a SRTM 90m DEM tile from OpenTopography (free, no login)
  3. Computes HAND using the pysheds hydrological pipeline
  4. Reprojects + resamples HAND to exactly match the S1 chip geometry
  5. Saves as data/sen1floods11/hand_chips/{chip_id}_HAND.tif

Why this matters:
  HAND (metres above nearest drainage) is the physics prior used by the
  HAND-gated attention model (Variants C and D). Without real HAND values
  the gate has no spatial signal, which eliminates the scientific contribution.

Install required packages (once, on DKUCC login node):
  conda activate terrainflood
  pip install pysheds requests

Usage:
  conda activate terrainflood
  python generate_hand_chips.py --data_root data/sen1floods11

  # Resume after interruption (skips already-created chips):
  python generate_hand_chips.py --data_root data/sen1floods11

  # Force regeneration of all chips:
  python generate_hand_chips.py --data_root data/sen1floods11 --overwrite

Expected runtime: ~45-90 minutes for 446 chips (network-bound, not GPU).
Run on the DKUCC LOGIN NODE (not via sbatch — needs internet access).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DEM download  (OpenTopography — free, no API key for SRTM GL3 90 m)
# ─────────────────────────────────────────────────────────────────────────────

def download_srtm_opentopo(
    west: float, south: float, east: float, north: float,
    out_path: Path,
    margin: float = 0.05,
    max_retries: int = 3,
) -> None:
    """
    Download SRTM GL3 (90 m) DEM from OpenTopography REST API.

    Args:
        west, south, east, north: bounding box in WGS84 degrees
        out_path: where to save the GeoTIFF
        margin:   buffer added around the bbox to avoid edge effects in HAND
        max_retries: number of retry attempts on network failure
    """
    import requests

    w = west  - margin
    s = south - margin
    e = east  + margin
    n = north + margin

    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=SRTMGL3"
        f"&south={s:.5f}&north={n:.5f}"
        f"&west={w:.5f}&east={e:.5f}"
        f"&outputFormat=GTiff"
    )

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and len(r.content) > 1000:
                out_path.write_bytes(r.content)
                return
            else:
                raise RuntimeError(
                    f"HTTP {r.status_code}  ({len(r.content)} bytes) — "
                    f"{r.text[:120]}"
                )
        except Exception as exc:
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"    attempt {attempt} failed ({exc}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HAND computation  (pysheds pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hand_from_dem(dem_path: Path) -> Tuple[np.ndarray, object, CRS]:
    """
    Run the standard HAND pipeline on a GeoTIFF DEM.

    Pipeline:
      fill_pits → fill_depressions → resolve_flats
      → flow_direction (D8) → hand

    Returns:
        hand_arr  : (H, W) float32 array, values in metres
        transform : affine transform of the output raster
        crs       : CRS of the output raster (matches DEM input)
    """
    from pysheds.grid import Grid

    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))

    # Hydrological conditioning
    pit_filled  = grid.fill_pits(dem)
    flooded     = grid.fill_depressions(pit_filled)
    inflated    = grid.resolve_flats(flooded)

    # Flow direction (D8) and HAND
    fdir = grid.flowdir(inflated)
    hand = grid.hand(fdir, inflated, inplace=False)

    hand_arr = np.asarray(hand, dtype=np.float32)
    hand_arr = np.nan_to_num(hand_arr, nan=0.0, posinf=100.0, neginf=0.0)
    hand_arr = np.clip(hand_arr, 0.0, 300.0)   # sensible physical range

    # Recover the affine transform from the grid
    transform = rasterio.transform.from_bounds(
        grid.bbox[0], grid.bbox[1], grid.bbox[2], grid.bbox[3],
        hand_arr.shape[1], hand_arr.shape[0],
    )
    crs = CRS.from_wkt(grid.crs.to_wkt())

    return hand_arr, transform, crs


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Reproject + save aligned to the S1 chip
# ─────────────────────────────────────────────────────────────────────────────

def save_hand_chip(
    hand_arr: np.ndarray,
    hand_transform,
    hand_crs: CRS,
    reference_s1: Path,
    output_path: Path,
) -> None:
    """
    Reproject hand_arr to exactly match the reference S1 chip geometry
    and write as a single-band float32 GeoTIFF.
    """
    with rasterio.open(reference_s1) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_height    = ref.height
        dst_width     = ref.width

    hand_reproj = np.zeros((dst_height, dst_width), dtype=np.float32)

    reproject(
        source        = hand_arr,
        destination   = hand_reproj,
        src_transform = hand_transform,
        src_crs       = hand_crs,
        dst_transform = dst_transform,
        dst_crs       = dst_crs,
        resampling    = Resampling.bilinear,
    )

    hand_reproj = np.clip(hand_reproj, 0.0, 300.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver    = "GTiff",
        height    = dst_height,
        width     = dst_width,
        count     = 1,
        dtype     = np.float32,
        crs       = dst_crs,
        transform = dst_transform,
        nodata    = -9999.0,
        compress  = "lzw",
    ) as dst:
        dst.write(hand_reproj, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-chip HAND GeoTIFFs for Sen1Floods11"
    )
    ap.add_argument(
        "--data_root", type=str, default="data/sen1floods11",
        help="Root of Sen1Floods11 dataset (same as used in training)"
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate chips that already exist (default: skip)"
    )
    args = ap.parse_args()

    # ── Check pysheds is installed ────────────────────────────
    try:
        import pysheds  # noqa: F401
    except ImportError:
        print(
            "\nERROR: pysheds is not installed.\n"
            "Install it with:\n"
            "  pip install pysheds\n"
            "Then re-run this script.\n"
        )
        sys.exit(1)

    try:
        import requests  # noqa: F401
    except ImportError:
        print(
            "\nERROR: requests is not installed.\n"
            "Install it with:\n"
            "  pip install requests\n"
        )
        sys.exit(1)

    data_root = Path(args.data_root)
    s1_dir    = data_root / "flood_events" / "HandLabeled" / "S1Hand"
    hand_dir  = data_root / "hand_chips"

    if not s1_dir.exists():
        print(f"ERROR: S1 directory not found: {s1_dir}")
        sys.exit(1)

    s1_files = sorted(s1_dir.glob("*_S1Hand.tif"))
    if not s1_files:
        print(f"ERROR: No S1Hand.tif files found in {s1_dir}")
        sys.exit(1)

    hand_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print("  Generate HAND chips for Sen1Floods11")
    print(f"  S1 chips found : {len(s1_files)}")
    print(f"  Output dir     : {hand_dir}")
    print("=" * 62)

    ok = err = skip = 0
    t_start = time.time()

    for i, s1_path in enumerate(s1_files):
        chip_id  = s1_path.stem.replace("_S1Hand", "")
        out_path = hand_dir / f"{chip_id}_HAND.tif"

        if out_path.exists() and not args.overwrite:
            skip += 1
            continue

        print(f"\n[{i+1:03d}/{len(s1_files)}] {chip_id}")

        # ── Read chip footprint ───────────────────────────────
        try:
            with rasterio.open(s1_path) as src:
                west, south, east, north = transform_bounds(
                    src.crs, CRS.from_epsg(4326),
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                )
        except Exception as exc:
            print(f"  FAIL  read bounds: {exc}")
            err += 1
            continue

        # ── Download SRTM DEM ─────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp_dem = Path(f.name)

        try:
            download_srtm_opentopo(west, south, east, north, tmp_dem)
        except Exception as exc:
            print(f"  FAIL  DEM download: {exc}")
            err += 1
            tmp_dem.unlink(missing_ok=True)
            continue

        # ── Compute HAND ──────────────────────────────────────
        try:
            hand_arr, hand_transform, hand_crs = compute_hand_from_dem(tmp_dem)
        except Exception as exc:
            print(f"  FAIL  HAND compute: {exc}")
            err += 1
            tmp_dem.unlink(missing_ok=True)
            continue
        finally:
            tmp_dem.unlink(missing_ok=True)

        # ── Reproject + save ──────────────────────────────────
        try:
            save_hand_chip(hand_arr, hand_transform, hand_crs, s1_path, out_path)
            hmin = hand_arr.min()
            hmax = hand_arr.max()
            hmean = hand_arr.mean()
            print(
                f"  OK    HAND range [{hmin:.1f}, {hmax:.1f}] m  "
                f"mean={hmean:.1f} m  → {out_path.name}"
            )
            ok += 1
        except Exception as exc:
            print(f"  FAIL  save: {exc}")
            err += 1

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 62)
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Generated : {ok}")
    print(f"  Skipped   : {skip}  (already existed)")
    print(f"  Failed    : {err}")
    print(f"  Output    : {hand_dir}")
    print("=" * 62)

    if err > 0:
        print(f"\n  WARNING: {err} chips failed. Re-run to retry them.")
        print("  (Already-generated chips will be skipped automatically)")

    if ok + skip == len(s1_files):
        print("\n  ALL chips have HAND files. Ready to retrain Variants C and D.")
    else:
        missing = len(s1_files) - ok - skip - err
        if missing > 0:
            print(f"\n  {missing} chips still missing HAND. Re-run this script.")


if __name__ == "__main__":
    main()
