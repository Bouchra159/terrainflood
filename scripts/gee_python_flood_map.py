"""
gee_python_flood_map.py  –  Google Earth Engine Python API flood map
====================================================================
Author : Bouchra Daddaoui | Duke Kunshan University | Signature Work 2026
Mentor : Prof. Dongmian Zou, Ph.D.

PURPOSE:
  Pull real Sentinel-1, Landsat 8, HAND, JRC, and WorldPop data for
  the Bolivia 2018 Amazon flood using the GEE Python API.
  Exports publication-quality static maps as PNGs + GeoTIFFs.

REQUIRES:
  pip install earthengine-api geemap matplotlib numpy pillow

AUTHENTICATE (first time):
  earthengine authenticate
  (opens a browser, log in with your Google account that has GEE access)

USAGE:
  python scripts/gee_python_flood_map.py

OUTPUT:
  results/gee_maps/
    bolivia_s1_preflood_vv.png
    bolivia_s1_postflood_vv.png
    bolivia_s1_change_vv.png
    bolivia_landsat8_truecolor.png
    bolivia_landsat8_falsecolor.png
    bolivia_hand_meters.png
    bolivia_hand_gate_alpha.png
    bolivia_flood_extent.png
    bolivia_population_exposure.png
    bolivia_composite_poster.png   ← main paper/poster figure
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path

OUT = Path("results/gee_maps")
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour maps ───────────────────────────────────────────────────────────────
FLOOD_CMAP = LinearSegmentedColormap.from_list(
    "flood", ["#FFFFFF", "#AED6F1", "#2980B9", "#1A5276", "#0B2545"])
HAND_CMAP  = LinearSegmentedColormap.from_list(
    "hand",  ["#08519c","#6baed6","#bdd7e7","#eff3ff",
               "#ffffcc","#fecc5c","#fd8d3c","#f03b20","#bd0026"])
GATE_CMAP  = LinearSegmentedColormap.from_list(
    "gate",  ["#bd0026","#fd8d3c","#fecc5c","#d9f0a3","#238b45"])
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "change",["#08306b","#2171b5","#6baed6","#f7f7f7","#fd8d3c","#d94801","#7f2704"])
SAR_CMAP = "gray"

TEAL = "#2E7873"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Study area ────────────────────────────────────────────────────────────────
# Beni Department, Bolivia — 2018 Amazon Flood
# Approximate bounds: [-67.8, -16.5, -62.8, -10.2] (lon_min, lat_min, lon_max, lat_max)
BBOX = (-67.8, -16.5, -62.8, -10.2)   # W, S, E, N

def get_ee():
    """Initialise the Earth Engine API. Returns the ee module."""
    try:
        import ee
        try:
            ee.Initialize()
        except Exception:
            print("Authenticating with Google Earth Engine…")
            ee.Authenticate()
            ee.Initialize()
        return ee
    except ModuleNotFoundError:
        print("ERROR: earthengine-api not installed.")
        print("  Run:  pip install earthengine-api")
        sys.exit(1)


def ee_to_numpy(image, bands, region, scale=100, vis_params=None):
    """
    Download an EE image as a numpy array using getThumbURL thumbnail.
    Returns (array, shape) where array is HxWxC uint8.
    """
    import requests
    from PIL import Image as PILImage
    from io import BytesIO

    url = image.getThumbURL({
        "bands": bands,
        "region": region,
        "dimensions": 512,
        "format": "png",
        **(vis_params or {}),
    })
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = PILImage.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def build_map():
    """Main function: pull GEE data and save publication maps."""
    ee = get_ee()

    roi = ee.Geometry.Rectangle(list(BBOX))

    print("Loading Sentinel-1 pre-flood (Jul-Nov 2017)…")
    s1pre = (ee.ImageCollection("COPERNICUS/S1_GRD")
             .filterBounds(roi)
             .filterDate("2017-07-01", "2017-11-30")
             .filter(ee.Filter.eq("instrumentMode", "IW"))
             .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
             .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
             .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
             .select(["VV", "VH"])
             .mean()
             .clip(roi))

    print("Loading Sentinel-1 post-flood (Jan-Mar 2018)…")
    s1post = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(roi)
              .filterDate("2018-01-01", "2018-03-31")
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
              .select(["VV", "VH"])
              .mean()
              .clip(roi))

    vv_change = s1post.select("VV").subtract(s1pre.select("VV"))
    flood_otsu = vv_change.lt(-3.0).And(s1post.select("VV").lt(-14.0))

    print("Loading HAND (MERIT Hydro)…")
    hand = (ee.Image("MERIT/Hydro/v1_0_1")
              .select("hnd")
              .clip(roi))
    gate_alpha = hand.multiply(-1.0 / 50.0).exp().rename("gate_alpha")

    print("Loading Landsat 8 (Jan-Apr 2018)…")
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
          .filterBounds(roi)
          .filterDate("2018-01-01", "2018-04-30")
          .filter(ee.Filter.lt("CLOUD_COVER", 30))
          .map(lambda img: img.updateMask(
              img.select("QA_PIXEL").bitwiseAnd(1 << 3).eq(0)))
          .median()
          .clip(roi))

    print("Loading JRC water…")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi)
    perm_water = jrc.select("occurrence").gt(80)

    print("Loading WorldPop 2018…")
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop")
           .filter(ee.Filter.eq("country", "BOL"))
           .filter(ee.Filter.eq("year", 2018))
           .first()
           .clip(roi))
    flood_pop = pop.updateMask(flood_otsu.And(perm_water.Not()))

    # ── Download thumbnails ───────────────────────────────────────────────────
    vis_sar   = {"min": -25, "max": 0}
    vis_l8tc  = {"bands": ["B4", "B3", "B2"], "min": 0.01, "max": 0.25, "gamma": 1.3}
    vis_l8fc  = {"bands": ["B5", "B4", "B3"], "min": 0.01, "max": 0.35, "gamma": 1.3}
    vis_hand  = {"min": 0, "max": 30, "palette": ["#08519c","#bdd7e7","#ffffcc","#f03b20","#bd0026"]}
    vis_gate  = {"min": 0, "max": 1,  "palette": ["#bd0026","#fecc5c","#d9f0a3","#238b45"]}
    vis_flood = {"min": 0, "max": 1,  "palette": ["white", "#08519c"]}
    vis_change= {"min": -8, "max": 2,  "palette": ["#08306b","#2171b5","#f7f7f7","#fd8d3c","#7f2704"]}
    vis_pop   = {"min": 0, "max": 20, "palette": ["#ffffcc","#fe9929","#cc4c02"]}

    layers = {
        "s1_pre_vv":        (s1pre.select("VV"),          ["VV"],           vis_sar,    "SAR VV Pre-flood (Jul–Nov 2017)"),
        "s1_post_vv":       (s1post.select("VV"),         ["VV"],           vis_sar,    "SAR VV Post-flood (Jan–Mar 2018)"),
        "s1_change_vv":     (vv_change,                   ["VV"],           vis_change, "SAR ΔVV (post − pre, dB)"),
        "l8_truecolor":     (l8,                          ["B4","B3","B2"], vis_l8tc,   "Landsat 8 True Colour (2018)"),
        "l8_falsecolor":    (l8,                          ["B5","B4","B3"], vis_l8fc,   "Landsat 8 False Colour NIR (2018)"),
        "hand":             (hand,                        ["hnd"],          vis_hand,   "HAND — Height Above Nearest Drainage (m)"),
        "gate_alpha":       (gate_alpha,                  ["gate_alpha"],   vis_gate,   "HAND Gate α = exp(−h/50)"),
        "flood_extent":     (flood_otsu.selfMask(),        None,            vis_flood,  "Flood Extent — Otsu Detection"),
        "pop_exposure":     (flood_pop,                   None,             vis_pop,    "Population in Flood Zone (WorldPop 2018)"),
    }

    arrays = {}
    for key, (img, bands, vis, title) in layers.items():
        print(f"  Downloading {title}…")
        try:
            arr = ee_to_numpy(img, bands, roi.getInfo()["coordinates"], vis_params=vis)
            arrays[key] = arr
            # Save individual PNG
            fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0D1117")
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
            ax.text(0.01, 0.01,
                    "TerrainFlood-UQ | Bolivia 2018 | GEE",
                    transform=ax.transAxes, fontsize=7.5, color="#AED6F1",
                    va="bottom", ha="left")
            plt.savefig(OUT / f"{key}.png", facecolor="#0D1117")
            plt.close()
            print(f"    Saved {key}.png")
        except Exception as e:
            print(f"    WARNING: could not download {title}: {e}")
            arrays[key] = None

    # ── Main composite figure (6-panel) ──────────────────────────────────────
    panel_order = [
        ("l8_truecolor",  "A  Landsat 8 True Colour\n(pre/post 2018)"),
        ("s1_post_vv",    "B  Sentinel-1 VV\n(post-flood Jan–Mar 2018)"),
        ("s1_change_vv",  "C  SAR Backscatter Change\n(ΔVV = post − pre)"),
        ("hand",          "D  HAND — Terrain Height\n(MERIT Hydro, metres)"),
        ("gate_alpha",    "E  HAND Gate  α = exp(−h/50)\n[TerrainFlood-UQ model prior]"),
        ("flood_extent",  "F  Flood Extent Detection\n(Otsu threshold on ΔVV)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="#0D1117")
    fig.subplots_adjust(hspace=0.08, wspace=0.05, top=0.90, bottom=0.04,
                        left=0.01, right=0.99)

    panel_colors = [TEAL, "#2C3E50", "#641E16", "#154360", "#1B5E20", "#2C3E50"]

    for ax, (key, title), bg in zip(axes.flat, panel_order, panel_colors):
        arr = arrays.get(key)
        if arr is not None:
            ax.imshow(arr, interpolation="bilinear")
        else:
            ax.imshow(np.zeros((512, 512, 3), dtype=np.uint8))
            ax.text(0.5, 0.5, "GEE data\nnot available\n(run with GEE auth)",
                    ha="center", va="center", color="white",
                    transform=ax.transAxes, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, color="white", fontsize=10, fontweight="bold",
                     pad=5, linespacing=1.4,
                     bbox=dict(boxstyle="round,pad=0.4", fc=bg, ec="none", alpha=0.8))

    fig.suptitle(
        "Bolivia 2018 Amazon Flood  ·  Google Earth Engine Multi-Source Analysis\n"
        "Sentinel-1 SAR + Landsat 8 + MERIT HAND + WorldPop  ·  TerrainFlood-UQ",
        fontsize=13, fontweight="bold", color="white", y=0.97, linespacing=1.5)

    fig.text(0.5, 0.01,
             "BBOX: 67.8°W–62.8°W, 16.5°S–10.2°S (Beni Dept., Bolivia)  "
             "·  GEE data: Copernicus, USGS, MERIT, WorldPop  "
             "·  Analysis: Daddaoui (2026), DKU Signature Work",
             ha="center", fontsize=8, color="#888888")

    plt.savefig(OUT / "bolivia_composite_gee.png", facecolor="#0D1117")
    plt.close()
    print(f"\nSaved main composite: {OUT}/bolivia_composite_gee.png")

    return arrays


if __name__ == "__main__":
    print("TerrainFlood-UQ — GEE Python Map Generator")
    print("=" * 55)
    print(f"Output: {OUT}\n")

    # Check if ee is available
    try:
        import ee
        ee.Initialize()
        print("GEE authenticated. Pulling real satellite data…\n")
        build_map()
    except Exception as e:
        print(f"GEE not authenticated: {e}")
        print("\nTo run with real GEE data:")
        print("  1. Install:  pip install earthengine-api geemap")
        print("  2. Auth:     earthengine authenticate")
        print("  3. Re-run:   python scripts/gee_python_flood_map.py")
        print("\nAlternatively, run the JavaScript version in GEE Code Editor:")
        print("  scripts/gee_bolivia_flood_maps.js")
        print("\nGenerating placeholder figure showing expected GEE outputs…")

        # Generate placeholder showing expected figure layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="#0D1117")
        fig.subplots_adjust(hspace=0.08, wspace=0.05, top=0.90, bottom=0.04)

        titles = [
            ("A", "Landsat 8 True Colour\n2018 Flood Season",       TEAL),
            ("B", "Sentinel-1 VV Post-flood\nJan–Mar 2018",          "#2C3E50"),
            ("C", "SAR ΔVV Change\npost − pre (dB)",                 "#641E16"),
            ("D", "HAND Terrain Height\nMERIT Hydro (metres)",        "#154360"),
            ("E", "HAND Gate α=exp(−h/50)\nTerrainFlood-UQ prior",   "#1B5E20"),
            ("F", "Flood Extent (Otsu)\nSen1Floods11 Bolivia chips", "#2C3E50"),
        ]
        for ax, (lbl, title, bg) in zip(axes.flat, titles):
            ax.set_facecolor("#1A252F")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.55, f"GEE Layer {lbl}", ha="center", va="center",
                    color="#5D6D7E", fontsize=20, fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.5, 0.35, "Run:\nearthengine authenticate\npython scripts/gee_python_flood_map.py",
                    ha="center", va="center", color="#888888", fontsize=8,
                    transform=ax.transAxes, fontfamily="monospace")
            ax.set_title(title, color="white", fontsize=10, fontweight="bold",
                         pad=5, linespacing=1.4,
                         bbox=dict(boxstyle="round,pad=0.4", fc=bg, ec="none", alpha=0.8))

        fig.suptitle(
            "Bolivia 2018 Amazon Flood  ·  GEE Multi-Source Analysis [PLACEHOLDER]\n"
            "Authenticate with earthengine-api to generate with real satellite data",
            fontsize=13, fontweight="bold", color="white", y=0.97, linespacing=1.5)

        plt.savefig(OUT / "bolivia_composite_gee_placeholder.png", facecolor="#0D1117")
        plt.close()
        print(f"Saved placeholder: {OUT}/bolivia_composite_gee_placeholder.png")
