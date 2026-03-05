#!/usr/bin/env python3
"""
generate_hand_chips.py
======================
Generate per-chip HAND (Height Above Nearest Drainage) GeoTIFFs
for all 446 Sen1Floods11 HandLabeled chips.

DEM source  : Copernicus GLO-30 (30 m) on AWS S3 — FREE, no API key.
HAND engine : numpy + scipy only  (no pysheds, no numba, no GDAL CLI).

Algorithm:
  1. Download Copernicus GLO-30 tile(s) covering the chip (cached locally)
  2. Fill nodata / ocean pixels with nearest valid elevation
  3. Fill depressions (iterative 3×3 minimum filter)
  4. D8 flow direction (steepest 8-neighbour descent)
  5. D8 flow accumulation (topological order, high → low)
  6. Stream mask  (flow accumulation > threshold)
  7. HAND = DEM[pixel] − DEM[nearest stream pixel]  (via scipy EDT)
  8. Reproject + resample to exactly match the S1 chip grid

Dependencies (all already in terrainflood env):
  rasterio, numpy, scipy, requests

Usage (run on DKUCC LOGIN NODE — needs internet; NOT via sbatch):
  conda activate terrainflood
  cd ~/terrainflood
  python generate_hand_chips.py --data_root data/sen1floods11

Resume interrupted run (skips already-created chips automatically):
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
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling, transform_bounds


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Copernicus GLO-30 tile download (AWS S3, free, no key)
# ─────────────────────────────────────────────────────────────────────────────

_COG_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_url(lat_floor: int, lon_floor: int) -> str:
    """URL of one 1°×1° Copernicus GLO-30 tile."""
    ns   = "N" if lat_floor >= 0 else "S"
    ew   = "E" if lon_floor >= 0 else "W"
    lat  = abs(lat_floor)
    lon  = abs(lon_floor)
    name = f"Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"
    return f"{_COG_BASE}/{name}/{name}.tif"


def _tiles_needed(
    west: float, south: float, east: float, north: float
) -> List[Tuple[int, int]]:
    """All (lat_floor, lon_floor) pairs that cover the bbox."""
    return [
        (lat, lon)
        for lat in range(int(np.floor(south)), int(np.floor(north)) + 1)
        for lon in range(int(np.floor(west)),  int(np.floor(east))  + 1)
    ]


def _download_tile(
    lat_floor: int, lon_floor: int,
    cache_dir: Path,
    max_retries: int = 3,
) -> Optional[Path]:
    """
    Download one tile to cache_dir and return its path.
    Returns None for ocean/missing tiles (HTTP 403/404).
    """
    import requests

    url   = _tile_url(lat_floor, lon_floor)
    fname = url.split("/")[-1]
    dst   = cache_dir / fname

    if dst.exists():
        return dst

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                dst.write_bytes(r.content)
                return dst
            if r.status_code in (403, 404):
                return None          # tile doesn't exist (ocean)
            raise RuntimeError(f"HTTP {r.status_code}")
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                raise RuntimeError(
                    f"Tile lat={lat_floor} lon={lon_floor} failed: {exc}"
                )
    return None


def get_dem_for_chip(
    west: float, south: float, east: float, north: float,
    cache_dir: Path,
    margin: float = 0.05,
) -> Optional[Path]:
    """
    Download and mosaic all Copernicus tiles covering the chip bbox.
    Returns a GeoTIFF path (temp mosaic OR cached single tile).
    Returns None if no land tiles exist (pure ocean chip).
    Mosaic files are marked with '_mosaic' and must be deleted by caller.
    """
    w, s, e, n = west - margin, south - margin, east + margin, north + margin
    tile_paths = [
        p for lat, lon in _tiles_needed(w, s, e, n)
        if (p := _download_tile(lat, lon, cache_dir)) is not None
    ]

    if not tile_paths:
        return None

    if len(tile_paths) == 1:
        return tile_paths[0]        # single tile — no mosaic needed

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, mosaic_tf = rio_merge(datasets, bounds=(w, s, e, n))
    finally:
        for ds in datasets:
            ds.close()

    tmp = Path(tempfile.mktemp(suffix="_mosaic.tif"))
    profile = {
        "driver": "GTiff", "count": 1,
        "height": mosaic.shape[1], "width": mosaic.shape[2],
        "dtype": mosaic.dtype,
        "crs": datasets[0].crs,
        "transform": mosaic_tf,
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
    return tmp


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HAND computation  (numpy + scipy — no pysheds)
# ─────────────────────────────────────────────────────────────────────────────

def _fill_nodata(dem: np.ndarray, nodata_val: float) -> np.ndarray:
    """Replace nodata / ocean pixels with nearest valid elevation."""
    from scipy.ndimage import distance_transform_edt

    invalid = (np.abs(dem - nodata_val) < 1) | np.isnan(dem) | (dem < -1000)
    if not invalid.any():
        return dem
    if invalid.all():
        return np.zeros_like(dem)

    _, idx = distance_transform_edt(invalid, return_indices=True)
    out = dem.copy()
    out[invalid] = dem[idx[0][invalid], idx[1][invalid]]
    return out


def _fill_depressions(dem: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """
    Iterative depression filling using a 3×3 minimum filter (no centre).
    Each iteration raises sink pixels to their lowest neighbour's elevation.
    Converges when no sinks remain.
    """
    from scipy.ndimage import minimum_filter

    fp = np.ones((3, 3), dtype=bool)
    fp[1, 1] = False          # exclude centre — gives true neighbour min

    filled = dem.copy()
    for _ in range(max_iter):
        min_nbr = minimum_filter(filled, footprint=fp, mode="nearest")
        sinks   = filled < min_nbr
        if not sinks.any():
            break
        filled[sinks] = min_nbr[sinks]
    return filled


def _d8_accumulation(dem: np.ndarray) -> np.ndarray:
    """
    D8 flow accumulation.

    For each pixel, finds the steepest downhill neighbour (8-connected),
    then accumulates counts from highest to lowest elevation.

    Returns an array where high values indicate main channels / rivers.
    """
    h, w = dem.shape

    # 8-neighbour offsets and diagonal distances
    di    = np.array([-1, -1,  0,  1,  1,  1,  0, -1])
    dj    = np.array([ 0,  1,  1,  1,  0, -1, -1, -1])
    ddist = np.array([1., np.sqrt(2), 1., np.sqrt(2),
                      1., np.sqrt(2), 1., np.sqrt(2)])

    padded = np.pad(dem, 1, mode="edge")

    # Vectorised flow direction: steepest descent wins
    best_dir   = np.full((h, w), -1, dtype=np.int8)
    best_slope = np.zeros((h, w), dtype=np.float32)  # positive slopes only

    for d in range(8):
        nbr   = padded[1 + di[d]:1 + di[d] + h,
                       1 + dj[d]:1 + dj[d] + w]
        slope = (dem - nbr) / ddist[d]
        mask  = slope > best_slope
        best_slope[mask] = slope[mask]
        best_dir[mask]   = d

    # Accumulate: iterate pixels from highest to lowest elevation
    acc   = np.ones((h, w), dtype=np.float32)
    order = np.argsort(dem.ravel())[::-1]      # descending elevation

    for flat_idx in order:
        r, c = divmod(int(flat_idx), w)
        d = int(best_dir[r, c])
        if d < 0:
            continue
        tr, tc = r + int(di[d]), c + int(dj[d])
        if 0 <= tr < h and 0 <= tc < w:
            acc[tr, tc] += acc[r, c]

    return acc


def compute_hand(dem_path: Path) -> Tuple[np.ndarray, object, CRS]:
    """
    Full HAND pipeline on a GeoTIFF DEM.  Returns (hand_arr, transform, crs).

    Steps:
      nodata fill → depression fill → D8 accumulation
      → stream mask → nearest-stream HAND via EDT
    """
    from scipy.ndimage import distance_transform_edt

    with rasterio.open(dem_path) as src:
        dem_raw   = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata if src.nodata is not None else -32768.0

    # 1. Fill nodata
    dem = _fill_nodata(dem_raw, nodata)

    if dem.max() == dem.min():
        # Flat/featureless chip (e.g. ocean tile) → return zeros
        return np.zeros_like(dem), transform, crs

    # 2. Fill depressions
    dem_f = _fill_depressions(dem)

    # 3. D8 flow accumulation
    acc = _d8_accumulation(dem_f)

    # 4. Stream mask  (top 0.5% of accumulation, min 50 cells)
    threshold = float(max(50, np.percentile(acc, 99.5)))
    streams   = acc >= threshold
    if not streams.any():
        # Fallback: lowest 10th percentile of elevation
        streams = dem_f <= float(np.percentile(dem_f, 10))

    # 5. HAND via nearest stream pixel
    _, idx        = distance_transform_edt(~streams, return_indices=True)
    stream_elev   = dem_f[idx[0], idx[1]]
    hand          = np.maximum(0.0, dem_f - stream_elev)

    return hand.astype(np.float32), transform, crs


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
    """Reproject HAND to the S1 chip's CRS/grid and save as float32 GeoTIFF."""
    with rasterio.open(s1_path) as ref:
        dst_crs  = ref.crs
        dst_tf   = ref.transform
        dst_h    = ref.height
        dst_w    = ref.width

    hand_reproj = np.zeros((dst_h, dst_w), dtype=np.float32)

    reproject(
        source        = hand_arr,
        destination   = hand_reproj,
        src_transform = hand_transform,
        src_crs       = hand_crs,
        dst_transform = dst_tf,
        dst_crs       = dst_crs,
        resampling    = Resampling.bilinear,
        src_nodata    = None,
        dst_nodata    = 0.0,
    )

    hand_reproj = np.where(np.isfinite(hand_reproj), hand_reproj, 0.0)
    hand_reproj = np.clip(hand_reproj, 0.0, 300.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff", count=1, dtype=np.float32,
        height=dst_h, width=dst_w,
        crs=dst_crs, transform=dst_tf,
        nodata=-9999.0, compress="lzw",
    ) as dst:
        dst.write(hand_reproj, 1)


def _write_zero_hand(s1_path: Path, out_path: Path) -> None:
    """Write an all-zeros HAND chip (fallback for ocean / missing DEM)."""
    with rasterio.open(s1_path) as ref:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver="GTiff", count=1, dtype=np.float32,
            height=ref.height, width=ref.width,
            crs=ref.crs, transform=ref.transform,
            nodata=-9999.0, compress="lzw",
        ) as dst:
            dst.write(np.zeros((ref.height, ref.width), dtype=np.float32), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-chip HAND GeoTIFFs — numpy/scipy, no API key"
    )
    ap.add_argument("--data_root", default="data/sen1floods11")
    ap.add_argument("--overwrite",  action="store_true",
                    help="Regenerate chips that already exist")
    ap.add_argument("--margin",     type=float, default=0.05,
                    help="WGS84 buffer around each chip when fetching DEM (°)")
    args = ap.parse_args()

    # Dependency check
    try:
        import requests  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install requests")

    data_root = Path(args.data_root)
    s1_dir    = data_root / "flood_events" / "HandLabeled" / "S1Hand"
    hand_dir  = data_root / "hand_chips"
    cache_dir = data_root / "dem_cache"

    for d in (hand_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not s1_dir.exists():
        sys.exit(f"ERROR: S1 directory not found: {s1_dir}")

    s1_files = sorted(s1_dir.glob("*_S1Hand.tif"))
    if not s1_files:
        sys.exit(f"ERROR: no *_S1Hand.tif files in {s1_dir}")

    print("=" * 66)
    print("  HAND chip generation — Copernicus GLO-30 / numpy+scipy")
    print(f"  Chips   : {len(s1_files)}")
    print(f"  Output  : {hand_dir}")
    print(f"  Cache   : {cache_dir}")
    print("=" * 66)

    ok = err = skip = 0
    t0 = time.time()

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
            print(f"  FAIL  bounds: {exc}")
            err += 1
            continue

        # ── Download / mosaic DEM ─────────────────────────────
        dem_path = None
        is_tmp   = False
        try:
            dem_path = get_dem_for_chip(
                west, south, east, north,
                cache_dir=cache_dir, margin=args.margin,
            )
            if dem_path is None:
                print("  WARN  no DEM tiles (ocean?) — writing zeros")
                _write_zero_hand(s1_path, out_path)
                skip += 1
                continue
            is_tmp = "_mosaic" in dem_path.name
        except Exception as exc:
            print(f"  FAIL  DEM download: {exc}")
            err += 1
            continue

        # ── Compute HAND ──────────────────────────────────────
        try:
            hand_arr, hand_tf, hand_crs = compute_hand(dem_path)
        except Exception as exc:
            print(f"  FAIL  HAND compute: {exc}")
            err += 1
            continue
        finally:
            if is_tmp and dem_path is not None:
                dem_path.unlink(missing_ok=True)

        # ── Reproject + save ──────────────────────────────────
        try:
            save_hand_chip(hand_arr, hand_tf, hand_crs, s1_path, out_path)
            print(
                f"  OK    min={hand_arr.min():.1f}  "
                f"mean={hand_arr.mean():.1f}  "
                f"max={hand_arr.max():.1f} m"
            )
            ok += 1
        except Exception as exc:
            print(f"  FAIL  save: {exc}")
            err += 1

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 66)
    print(f"  Done in {elapsed / 60:.1f} min")
    print(f"  Generated : {ok}")
    print(f"  Skipped   : {skip}  (existed / ocean)")
    print(f"  Failed    : {err}")
    print("=" * 66)

    if err > 0:
        print(f"\n  {err} failed — re-run to retry (done chips are skipped).")
    if ok + skip >= len(s1_files) - err:
        print("\n  All chips done. Next step:")
        print("    sbatch jobs/train_C.sbatch")
        print("    sbatch jobs/train_D.sbatch")


if __name__ == "__main__":
    main()
