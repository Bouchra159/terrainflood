#!/usr/bin/env python3
"""
generate_hand_chips.py
======================
Generate per-chip HAND (Height Above Nearest Drainage) GeoTIFFs
for all 446 Sen1Floods11 HandLabeled chips.

DEM source : Copernicus GLO-30 (30 m resolution)
             Hosted on AWS S3 as public open data — NO API KEY REQUIRED.
             https://copernicus-dem-30m.s3.amazonaws.com/

HAND algorithm (pysheds):
    fill_pits → fill_depressions → resolve_flats
    → D8 flow direction → HAND

Output: data/sen1floods11/hand_chips/{chip_id}_HAND.tif
        Float32, same CRS / pixel grid as the matching S1 chip.

Install (once, on DKUCC login node):
    conda activate terrainflood
    pip install pysheds requests

Run (on LOGIN NODE — needs internet; NOT via sbatch):
    conda activate terrainflood
    cd ~/terrainflood
    python generate_hand_chips.py --data_root data/sen1floods11

Resume after interruption (skips already-done chips automatically):
    python generate_hand_chips.py --data_root data/sen1floods11

Force redo everything:
    python generate_hand_chips.py --data_root data/sen1floods11 --overwrite
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling, transform_bounds


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Copernicus GLO-30 tile download  (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

_COG_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_url(lat_floor: int, lon_floor: int) -> str:
    """
    Build the S3 HTTPS URL for a single 1°×1° Copernicus GLO-30 tile.

    Tile naming convention:
        Copernicus_DSM_COG_10_{NS}{LAT:02d}_00_{EW}{LON:03d}_00_DEM
    where NS = N/S and EW = E/W based on sign of floor coords.
    """
    ns  = "N" if lat_floor >= 0 else "S"
    ew  = "E" if lon_floor >= 0 else "W"
    lat = abs(lat_floor)
    lon = abs(lon_floor)
    name = f"Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"
    return f"{_COG_BASE}/{name}/{name}.tif"


def _tiles_for_bbox(
    west: float, south: float, east: float, north: float
) -> List[Tuple[int, int]]:
    """Return all (lat_floor, lon_floor) tiles that intersect the bbox."""
    lat_min = int(np.floor(south))
    lat_max = int(np.floor(north))
    lon_min = int(np.floor(west))
    lon_max = int(np.floor(east))
    return [
        (lat, lon)
        for lat in range(lat_min, lat_max + 1)
        for lon in range(lon_min, lon_max + 1)
    ]


def download_tile(
    lat_floor: int,
    lon_floor: int,
    cache_dir: Path,
    max_retries: int = 3,
) -> Path | None:
    """
    Download one Copernicus GLO-30 tile to cache_dir.
    Returns the local path, or None if the tile does not exist
    (e.g. open ocean tiles are not published).
    """
    import requests

    url = _tile_url(lat_floor, lon_floor)
    fname = url.split("/")[-1]
    dst = cache_dir / fname

    if dst.exists():
        return dst          # already cached

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                dst.write_bytes(r.content)
                return dst
            if r.status_code == 403:
                # 403 = tile does not exist (ocean / no data)
                return None
            raise RuntimeError(f"HTTP {r.status_code}")
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                raise RuntimeError(
                    f"Failed to download tile lat={lat_floor} lon={lon_floor}: {exc}"
                )
    return None


def get_dem_for_chip(
    west: float, south: float, east: float, north: float,
    cache_dir: Path,
    margin: float = 0.05,
) -> Path | None:
    """
    Download all Copernicus tiles covering the chip bbox (+ margin),
    mosaic them if needed, and return a temp GeoTIFF path.
    Caller is responsible for deleting the returned temp file.
    Returns None if no land tiles exist (pure ocean chip).
    """
    w, s, e, n = west - margin, south - margin, east + margin, north + margin
    tiles = _tiles_for_bbox(w, s, e, n)

    tile_paths = []
    for lat, lon in tiles:
        p = download_tile(lat, lon, cache_dir)
        if p is not None:
            tile_paths.append(p)

    if not tile_paths:
        return None     # no land tiles → pure ocean

    # Single tile: just return it directly (no mosaic needed)
    if len(tile_paths) == 1:
        return tile_paths[0]

    # Multiple tiles: mosaic and clip to bbox + margin
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, mosaic_transform = rio_merge(datasets, bounds=(w, s, e, n))
    finally:
        for ds in datasets:
            ds.close()

    # Write mosaic to a temp file
    tmp = Path(tempfile.mktemp(suffix="_mosaic.tif"))
    profile = datasets[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=mosaic_transform,
        count=1,
    )
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
    return tmp          # caller must unlink this


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HAND computation  (pysheds)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hand(dem_path: Path) -> Tuple[np.ndarray, object, CRS]:
    """
    Run the standard HAND pipeline on a GeoTIFF DEM.

    Pipeline:
        fill_pits → fill_depressions → resolve_flats
        → D8 flow direction → HAND (metres above nearest drainage)

    Returns:
        hand_arr  : (H, W) float32 clipped to [0, 300] m
        transform : affine transform of the output array (from rasterio)
        crs       : CRS of the output array (from rasterio)
    """
    from pysheds.grid import Grid

    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))

    pit_filled = grid.fill_pits(dem)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)
    fdir       = grid.flowdir(inflated)
    hand       = grid.hand(fdir, inflated, inplace=False)

    hand_arr = np.asarray(hand, dtype=np.float32)
    hand_arr = np.nan_to_num(hand_arr, nan=0.0, posinf=300.0, neginf=0.0)
    hand_arr = np.clip(hand_arr, 0.0, 300.0)

    # Read CRS and transform from rasterio (authoritative source)
    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs       = src.crs

    return hand_arr, transform, crs


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Reproject + save aligned to S1 chip
# ─────────────────────────────────────────────────────────────────────────────

def save_hand_chip(
    hand_arr: np.ndarray,
    hand_transform,
    hand_crs: CRS,
    s1_path: Path,
    out_path: Path,
) -> None:
    """
    Reproject hand_arr to exactly match the S1 chip's CRS and pixel grid,
    then write as a single-band float32 GeoTIFF.
    """
    with rasterio.open(s1_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_h         = ref.height
        dst_w         = ref.width

    hand_reproj = np.zeros((dst_h, dst_w), dtype=np.float32)

    reproject(
        source        = hand_arr,
        destination   = hand_reproj,
        src_transform = hand_transform,
        src_crs       = hand_crs,
        dst_transform = dst_transform,
        dst_crs       = dst_crs,
        resampling    = Resampling.bilinear,
        src_nodata    = -9999.0,
        dst_nodata    = 0.0,
    )

    # Replace nodata fill with 0 (sensible default: assume at drainage level)
    hand_reproj = np.where(np.isfinite(hand_reproj), hand_reproj, 0.0)
    hand_reproj = np.clip(hand_reproj, 0.0, 300.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver    = "GTiff",
        height    = dst_h,
        width     = dst_w,
        count     = 1,
        dtype     = np.float32,
        crs       = dst_crs,
        transform = dst_transform,
        nodata    = -9999.0,
        compress  = "lzw",
    ) as dst:
        dst.write(hand_reproj, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────────────────

def check_imports() -> None:
    """Exit early with a helpful message if required packages are missing."""
    missing = []
    try:
        import pysheds  # noqa: F401
    except ImportError:
        missing.append("pysheds")
    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")
    if missing:
        print(
            f"\nERROR: missing packages: {missing}\n"
            f"Install with:\n  pip install {' '.join(missing)}\n"
        )
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-chip HAND GeoTIFFs from Copernicus GLO-30 DEM"
    )
    ap.add_argument(
        "--data_root", type=str, default="data/sen1floods11",
        help="Root of Sen1Floods11 dataset",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate chips that already exist (default: skip)",
    )
    ap.add_argument(
        "--margin", type=float, default=0.05,
        help="WGS84 degree buffer around each chip when downloading DEM (default 0.05°)",
    )
    args = ap.parse_args()

    check_imports()

    data_root  = Path(args.data_root)
    s1_dir     = data_root / "flood_events" / "HandLabeled" / "S1Hand"
    hand_dir   = data_root / "hand_chips"
    cache_dir  = data_root / "dem_cache"   # DEM tiles reused across chips

    for d in (hand_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not s1_dir.exists():
        print(f"ERROR: S1 directory not found: {s1_dir}")
        sys.exit(1)

    s1_files = sorted(s1_dir.glob("*_S1Hand.tif"))
    if not s1_files:
        print(f"ERROR: no *_S1Hand.tif files found in {s1_dir}")
        sys.exit(1)

    print("=" * 64)
    print("  Generate HAND chips — Copernicus GLO-30  (no API key)")
    print(f"  S1 chips   : {len(s1_files)}")
    print(f"  Output     : {hand_dir}")
    print(f"  DEM cache  : {cache_dir}")
    print("=" * 64)

    ok = err = skip = 0
    t0 = time.time()

    for i, s1_path in enumerate(s1_files):
        chip_id  = s1_path.stem.replace("_S1Hand", "")
        out_path = hand_dir / f"{chip_id}_HAND.tif"

        if out_path.exists() and not args.overwrite:
            skip += 1
            continue

        print(f"\n[{i+1:03d}/{len(s1_files)}] {chip_id}")

        # ── 1. Read chip footprint in WGS84 ──────────────────
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

        # ── 2. Download / mosaic DEM tiles ────────────────────
        dem_path = None
        is_mosaic = False
        try:
            dem_path = get_dem_for_chip(
                west, south, east, north,
                cache_dir=cache_dir,
                margin=args.margin,
            )
            if dem_path is None:
                print("  WARN  no land tiles — chip may be over ocean; writing zeros")
                # Write a zeros chip so the model can still run
                _write_zero_hand(s1_path, out_path)
                skip += 1
                continue
            # If get_dem_for_chip returned a mosaic temp file, mark for cleanup
            is_mosaic = "_mosaic" in dem_path.name
        except Exception as exc:
            print(f"  FAIL  DEM download: {exc}")
            err += 1
            continue

        # ── 3. Compute HAND ───────────────────────────────────
        try:
            hand_arr, hand_transform, hand_crs = compute_hand(dem_path)
        except Exception as exc:
            print(f"  FAIL  HAND compute: {exc}")
            err += 1
            if is_mosaic:
                dem_path.unlink(missing_ok=True)
            continue
        finally:
            if is_mosaic and dem_path is not None:
                dem_path.unlink(missing_ok=True)

        # ── 4. Reproject + save ───────────────────────────────
        try:
            save_hand_chip(hand_arr, hand_transform, hand_crs, s1_path, out_path)
            print(
                f"  OK    [{hand_arr.min():.1f}, {hand_arr.mean():.1f}, "
                f"{hand_arr.max():.1f}] m  → {out_path.name}"
            )
            ok += 1
        except Exception as exc:
            print(f"  FAIL  save: {exc}")
            err += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f"  Finished in {elapsed / 60:.1f} min")
    print(f"  Generated : {ok}")
    print(f"  Skipped   : {skip}  (already existed / ocean)")
    print(f"  Failed    : {err}")
    print(f"  DEM tiles cached in: {cache_dir}")
    print("=" * 64)

    total_done = ok + skip
    if err > 0:
        print(f"\n  {err} chips failed — re-run to retry (already-done chips skipped).")
    if total_done >= len(s1_files) - err:
        print("\n  All chips complete. Ready to retrain Variants C and D.")
        print("  Run:")
        print("    sbatch jobs/train_C.sbatch")
        print("    sbatch jobs/train_D.sbatch")


def _write_zero_hand(s1_path: Path, out_path: Path) -> None:
    """Write a zeros HAND chip matching the S1 chip geometry (ocean fallback)."""
    with rasterio.open(s1_path) as ref:
        zeros = np.zeros((ref.height, ref.width), dtype=np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver="GTiff", height=ref.height, width=ref.width,
            count=1, dtype=np.float32,
            crs=ref.crs, transform=ref.transform,
            nodata=-9999.0, compress="lzw",
        ) as dst:
            dst.write(zeros, 1)


if __name__ == "__main__":
    main()
