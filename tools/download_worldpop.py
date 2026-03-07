#!/usr/bin/env python3
"""
WorldPop Population Chip Preparation
File: tools/download_worldpop.py

Downloads WorldPop 2020 country-level population density GeoTIFFs
(freely available, no API key or GEE access required) and crops/resamples
them to match each Sen1Floods11 chip's spatial extent and resolution.

Output: data/pop_chips/{chip_id}_pop.tif
  - Single-band float32 GeoTIFF
  - Units: people per 100m² pixel (WorldPop default)
  - CRS/transform aligned to corresponding S1Hand.tif chip
  - nodata = 0.0 (WorldPop uses -99999; remapped to 0)

Supported countries (extend WorldPop_URLS dict to add more):
  BOL — Bolivia  (test split,  15 chips)
  PRY — Paraguay (val split,   67 chips)
  GHA — Ghana    (train split, 53 chips)
  IND — India    (train split, 68 chips)
  NGA — Nigeria  (train split, 18 chips)
  PAK — Pakistan (train split, 28 chips)
  SOM — Somalia  (train split, 26 chips)
  ESP — Spain    (train split, 30 chips)
  LKA — Sri Lanka(train split, 42 chips)
  USA — USA      (train split, 69 chips)
  MMR — Mekong/Myanmar (train split, 30 chips)

Usage (Bolivia test chips only — most common):
  python tools/download_worldpop.py \\
      --data_root  data/sen1floods11 \\
      --output_dir data/pop_chips \\
      --events     Bolivia

Usage (val set — Paraguay):
  python tools/download_worldpop.py \\
      --data_root  data/sen1floods11 \\
      --output_dir data/pop_chips \\
      --events     Paraguay

Usage (all events — downloads multiple country rasters):
  python tools/download_worldpop.py \\
      --data_root  data/sen1floods11 \\
      --output_dir data/pop_chips \\
      --events     all

After running this script, check pop chips were created:
  ls data/pop_chips/ | head -20
  python tools/download_worldpop.py --check --data_root data/sen1floods11

Then run population exposure:
  python 06_exposure.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --pop_dir    data/pop_chips \\
      --output_dir results/exposure_D \\
      --T 20 --calibrate
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, calculate_default_transform


# ─────────────────────────────────────────────────────────────
# WorldPop 2020 country raster URLs (100m constrained)
# Source: https://www.worldpop.org/geodata/listing?id=29
# ─────────────────────────────────────────────────────────────

WORLDPOP_URLS: dict[str, str] = {
    "BOL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/BOL/bol_ppp_2020_constrained.tif",
    "PRY": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/PRY/pry_ppp_2020_constrained.tif",
    "GHA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/GHA/gha_ppp_2020_constrained.tif",
    "IND": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/IND/ind_ppp_2020_constrained.tif",
    "NGA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/NGA/nga_ppp_2020_constrained.tif",
    "PAK": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/PAK/pak_ppp_2020_constrained.tif",
    "SOM": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/SOM/som_ppp_2020_constrained.tif",
    "ESP": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/ESP/esp_ppp_2020_constrained.tif",
    "LKA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/LKA/lka_ppp_2020_constrained.tif",
    "USA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/USA/usa_ppp_2020_constrained.tif",
    "MMR": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/MMR/mmr_ppp_2020_constrained.tif",
    "KHM": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/KHM/khm_ppp_2020_constrained.tif",
    "COD": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/COD/cod_ppp_2020_constrained.tif",
    "CAN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/CAN/can_ppp_2020_constrained.tif",
    "ECU": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/ECU/ecu_ppp_2020_constrained.tif",
}

# Sen1Floods11 event → ISO3 country code
EVENT_TO_ISO: dict[str, str] = {
    "Bolivia":     "BOL",
    "Paraguay":    "PRY",
    "Ghana":       "GHA",
    "India":       "IND",
    "Nigeria":     "NGA",
    "Pakistan":    "PAK",
    "Somalia":     "SOM",
    "Spain":       "ESP",
    "Sri-Lanka":   "LKA",
    "USA":         "USA",
    "Mekong":      "MMR",   # chips are mostly in Myanmar/Mekong region
    "Cambodia":    "KHM",
    "DemRepCongo": "COD",
    "Canada":      "CAN",
    "Ecuador":     "ECU",
}

# WorldPop nodata value (varies by product)
WP_NODATA = -99999.0


# ─────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────

def _reporthook(count: int, block_size: int, total_size: int) -> None:
    """Simple progress reporter for urllib.request.urlretrieve."""
    if total_size <= 0:
        print(f"\r  Downloaded {count * block_size / 1e6:.1f} MB", end="", flush=True)
    else:
        pct = min(count * block_size / total_size * 100, 100)
        mb  = count * block_size / 1e6
        tot = total_size / 1e6
        print(f"\r  {pct:5.1f}%  {mb:.0f}/{tot:.0f} MB", end="", flush=True)


def download_country_raster(
    iso:       str,
    cache_dir: Path,
    force:     bool = False,
) -> Path | None:
    """
    Downloads WorldPop 2020 constrained population raster for the given
    ISO3 country code. Caches to cache_dir; skips download if already
    present unless force=True.

    Args:
        iso:       ISO3 country code (e.g. "BOL")
        cache_dir: directory to cache the downloaded raster
        force:     re-download even if file exists

    Returns:
        Path to the downloaded raster, or None on failure.
    """
    iso = iso.upper()
    if iso not in WORLDPOP_URLS:
        print(f"  [SKIP] No WorldPop URL configured for ISO={iso}")
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    url      = WORLDPOP_URLS[iso]
    filename = url.split("/")[-1]
    out_path = cache_dir / filename

    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [Cached] {out_path.name}  ({size_mb:.0f} MB)")
        return out_path

    print(f"  [Download] {iso}  ←  {url}")
    try:
        urllib.request.urlretrieve(url, out_path, reporthook=_reporthook)
        print()  # newline after progress bar
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved → {out_path}  ({size_mb:.0f} MB)")
        return out_path
    except urllib.error.URLError as e:
        print(f"\n  [ERROR] Download failed for {iso}: {e}")
        # Try unconstrained version as fallback
        fallback_url = (
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020/"
            f"2020/{iso}/{iso.lower()}_ppp_2020.tif"
        )
        print(f"  [Retry] Trying unconstrained version: {fallback_url}")
        try:
            fallback_path = cache_dir / f"{iso.lower()}_ppp_2020_unconstrained.tif"
            urllib.request.urlretrieve(fallback_url, fallback_path, reporthook=_reporthook)
            print()
            print(f"  Saved fallback → {fallback_path}")
            return fallback_path
        except urllib.error.URLError as e2:
            print(f"\n  [ERROR] Fallback also failed: {e2}")
            if out_path.exists():
                out_path.unlink()
            return None


# ─────────────────────────────────────────────────────────────
# Chip cropping
# ─────────────────────────────────────────────────────────────

def crop_worldpop_to_chip(
    worldpop_path: Path,
    chip_s1_path:  Path,
    out_path:      Path,
) -> bool:
    """
    Warps and crops the WorldPop country raster to exactly match the
    spatial extent and resolution of a Sen1Floods11 chip.

    Process:
      1. Read chip CRS, transform, height, width from S1Hand.tif
      2. Reproject WorldPop to chip CRS on-the-fly using rasterio VRT
      3. Resample to chip resolution using bilinear interpolation
      4. Clamp nodata pixels to 0.0 (no negative population)
      5. Write as float32 GeoTIFF with chip's CRS + transform

    Args:
        worldpop_path: path to downloaded country raster
        chip_s1_path:  path to {chip_id}_S1Hand.tif (reference)
        out_path:      output path for the population chip

    Returns:
        True on success, False on failure.
    """
    try:
        # Read chip spatial metadata
        with rasterio.open(chip_s1_path) as chip_src:
            chip_crs       = chip_src.crs
            chip_transform = chip_src.transform
            chip_height    = chip_src.height
            chip_width     = chip_src.width

        # Read and reproject WorldPop data to chip extent
        with rasterio.open(worldpop_path) as wp_src:
            wp_nodata = wp_src.nodata if wp_src.nodata is not None else WP_NODATA

            # Calculate destination transform for reprojection
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs    = wp_src.crs,
                dst_crs    = chip_crs,
                width      = chip_width,
                height     = chip_height,
                left       = chip_transform.c,
                bottom     = chip_transform.f + chip_transform.e * chip_height,
                right      = chip_transform.c + chip_transform.a * chip_width,
                top        = chip_transform.f,
            )

            # Allocate output array
            pop_data = np.zeros((chip_height, chip_width), dtype=np.float32)

            reproject(
                source       = rasterio.band(wp_src, 1),
                destination  = pop_data,
                src_transform = wp_src.transform,
                src_crs      = wp_src.crs,
                dst_transform = chip_transform,
                dst_crs      = chip_crs,
                resampling   = Resampling.bilinear,
                src_nodata   = wp_nodata,
                dst_nodata   = 0.0,
            )

        # Clean up: clamp negatives and nodata to 0
        pop_data = np.where(
            (pop_data < 0) | np.isnan(pop_data),
            0.0,
            pop_data,
        ).astype(np.float32)

        # Write output chip
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver    = "GTiff",
            height    = chip_height,
            width     = chip_width,
            count     = 1,
            dtype     = "float32",
            crs       = chip_crs,
            transform = chip_transform,
            nodata    = 0.0,
            compress  = "lzw",
        ) as dst:
            dst.write(pop_data, 1)

        # Quick sanity check
        total_pop = float(pop_data.sum())
        max_pop   = float(pop_data.max())
        return True, total_pop, max_pop

    except Exception as e:
        return False, 0.0, 0.0


# ─────────────────────────────────────────────────────────────
# Check existing chips
# ─────────────────────────────────────────────────────────────

def check_pop_chips(
    data_root:   Path,
    pop_dir:     Path,
) -> None:
    """
    Checks which Sen1Floods11 chips have population data and which don't.
    Prints a summary table.
    """
    s1_dir    = data_root / "flood_events" / "HandLabeled" / "S1Hand"
    all_chips = sorted(s1_dir.glob("*_S1Hand.tif"))

    print(f"\n{'='*60}")
    print(f"{'Chip ID':<40} {'Pop chip':>10} {'Total pop':>12}")
    print(f"{'─'*60}")

    has_pop    = 0
    missing    = 0
    by_event:  dict[str, dict] = {}

    for s1_path in all_chips:
        chip_id  = s1_path.stem.replace("_S1Hand", "")
        event    = chip_id.split("_")[0]
        pop_path = pop_dir / f"{chip_id}_pop.tif"

        if pop_path.exists():
            with rasterio.open(pop_path) as src:
                data = src.read(1)
            total = float(data.sum())
            status = f"✓  {total:10.0f}"
            has_pop += 1
            if event not in by_event:
                by_event[event] = {"chips": 0, "total_pop": 0.0}
            by_event[event]["chips"]     += 1
            by_event[event]["total_pop"] += total
        else:
            status = "✗  MISSING"
            missing += 1

        print(f"  {chip_id:<38} {status}")

    print(f"{'─'*60}")
    print(f"\nSummary:")
    print(f"  Has pop chip : {has_pop:4d}")
    print(f"  Missing      : {missing:4d}")
    print(f"  Total chips  : {len(all_chips):4d}")

    if by_event:
        print(f"\nBy event:")
        for ev, stats in sorted(by_event.items()):
            print(f"  {ev:<20} {stats['chips']:3d} chips  "
                  f"{stats['total_pop']:12,.0f} total population")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    pop_dir   = Path(args.output_dir)
    pop_dir.mkdir(parents=True, exist_ok=True)

    # Check mode only
    if args.check:
        check_pop_chips(data_root, pop_dir)
        return

    # Discover chips to process
    s1_dir    = data_root / "flood_events" / "HandLabeled" / "S1Hand"
    if not s1_dir.exists():
        raise FileNotFoundError(
            f"S1Hand directory not found: {s1_dir}\n"
            f"Make sure Sen1Floods11 is downloaded to {data_root}"
        )

    all_s1_chips = sorted(s1_dir.glob("*_S1Hand.tif"))
    if not all_s1_chips:
        raise RuntimeError(f"No *_S1Hand.tif files found in {s1_dir}")

    # Filter by requested events
    if args.events.lower() != "all":
        requested = {e.strip() for e in args.events.split(",")}
        chips_to_process = [
            p for p in all_s1_chips
            if p.stem.replace("_S1Hand", "").split("_")[0] in requested
        ]
    else:
        chips_to_process = all_s1_chips

    if not chips_to_process:
        print(f"No chips found for events: {args.events}")
        return

    print(f"Processing {len(chips_to_process)} chips "
          f"from events: {args.events}")

    # Determine which ISO codes we need
    needed_isos: set[str] = set()
    for s1_path in chips_to_process:
        event = s1_path.stem.replace("_S1Hand", "").split("_")[0]
        iso   = EVENT_TO_ISO.get(event)
        if iso:
            needed_isos.add(iso)
        else:
            print(f"  [WARN] No ISO mapping for event={event}, skipping")

    print(f"\nRequired country rasters: {sorted(needed_isos)}")

    # Download country rasters
    cache_dir = data_root / ".worldpop_cache"
    raster_map: dict[str, Path | None] = {}
    for iso in needed_isos:
        raster_map[iso] = download_country_raster(iso, cache_dir, force=args.force)

    # Crop to each chip
    print(f"\nCropping to {len(chips_to_process)} chips...")
    n_ok = 0
    n_skip = 0
    n_fail = 0
    summary: list[dict] = []

    for s1_path in chips_to_process:
        chip_id   = s1_path.stem.replace("_S1Hand", "")
        event     = chip_id.split("_")[0]
        iso       = EVENT_TO_ISO.get(event)
        out_path  = pop_dir / f"{chip_id}_pop.tif"

        if out_path.exists() and not args.force:
            n_skip += 1
            continue

        if iso is None or raster_map.get(iso) is None:
            print(f"  [SKIP]  {chip_id}  (no raster for {event}/{iso})")
            n_skip += 1
            continue

        ok, total_pop, max_pop = crop_worldpop_to_chip(
            worldpop_path = raster_map[iso],
            chip_s1_path  = s1_path,
            out_path      = out_path,
        )

        if ok:
            n_ok += 1
            summary.append({
                "chip_id":   chip_id,
                "event":     event,
                "iso":       iso,
                "total_pop": round(total_pop, 1),
                "max_pop":   round(max_pop, 2),
            })
            print(f"  [OK]    {chip_id:<38}  "
                  f"total={total_pop:10.0f}  max={max_pop:.2f}")
        else:
            n_fail += 1
            print(f"  [FAIL]  {chip_id}")

    # Save summary JSON
    summary_path = pop_dir / "pop_summary.json"
    summary_payload = {
        "n_ok":    n_ok,
        "n_skip":  n_skip,
        "n_fail":  n_fail,
        "chips":   summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    # Final report
    print(f"\n{'='*55}")
    print(f"  Created : {n_ok:4d} chips")
    print(f"  Skipped : {n_skip:4d} (already exist or no raster)")
    print(f"  Failed  : {n_fail:4d}")
    print(f"  Output  : {pop_dir}")
    print(f"  Summary : {summary_path}")
    print(f"{'='*55}")

    if n_ok > 0:
        print(f"\n✓ Population chips ready. Now run:")
        print(f"  python 06_exposure.py \\")
        print(f"      --checkpoint checkpoints/variant_D/best.pt \\")
        print(f"      --data_root  data/sen1floods11 \\")
        print(f"      --pop_dir    {pop_dir} \\")
        print(f"      --output_dir results/exposure_D \\")
        print(f"      --T 20 --calibrate")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download WorldPop and prepare per-chip population rasters"
    )
    p.add_argument("--data_root",   type=str, default="data/sen1floods11",
                   help="Path to Sen1Floods11 data root")
    p.add_argument("--output_dir",  type=str, default="data/pop_chips",
                   help="Output directory for population chip GeoTIFFs")
    p.add_argument("--events",      type=str, default="Bolivia",
                   help="Comma-separated event names, or 'all'. "
                        "E.g. 'Bolivia' or 'Bolivia,Paraguay'")
    p.add_argument("--force",       action="store_true",
                   help="Re-download and re-crop even if files exist")
    p.add_argument("--check",       action="store_true",
                   help="Check which chips have pop data (no download)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
