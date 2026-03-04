"""
Phase 1 — GEE Export Pipeline (FIXED v2)
==========================================
File: pipeline/01_gee_export.py

WHAT CHANGED:
  Original version used huge bounding boxes (Canada = 800x500 km at 10m).
  That is billions of pixels — hence 18+ hours and still running.

NEW APPROACH — two fast modes:
  1. stats  : computes HAND + WorldPop stats per event as CSV (no GEE tasks,
               runs in ~3 minutes entirely client-side)
  2. rasters: exports HAND at 30m resolution with TIGHT bboxes (~5-10 min each)

Usage:
  # Cancel all old stuck tasks first on code.earthengine.google.com/tasks

  # Step 1: get per-event stats CSV (fast, no tasks)
  python pipeline/01_gee_export.py --project fast-kiln-489119-e1 --mode stats

  # Step 2: export HAND rasters at 30m (fast tasks, ~10 min each)
  python pipeline/01_gee_export.py --project fast-kiln-489119-e1 --mode rasters --monitor
"""

import ee
import argparse
import json
import time
import csv
from pathlib import Path


FLOOD_EVENTS = {
    "Bolivia": {
        "pre_start":  "2018-01-15", "pre_end":  "2018-02-01",
        "post_start": "2018-02-01", "post_end": "2018-02-28",
        "bbox": [-65.5, -14.5, -64.0, -13.0],
        "split": "test"
    },
    "Cambodia": {
        "pre_start":  "2020-09-15", "pre_end":  "2020-10-01",
        "post_start": "2020-10-01", "post_end": "2020-10-20",
        "bbox": [104.5, 11.5, 106.0, 13.0],
        "split": "train"
    },
    "Canada": {
        "pre_start":  "2019-04-15", "pre_end":  "2019-05-01",
        "post_start": "2019-05-01", "post_end": "2019-05-20",
        "bbox": [-76.5, 46.5, -75.0, 47.5],
        "split": "train"
    },
    "DemRepCongo": {
        "pre_start":  "2019-10-15", "pre_end":  "2019-11-01",
        "post_start": "2019-11-01", "post_end": "2019-11-20",
        "bbox": [17.0, -4.5, 18.5, -3.0],
        "split": "train"
    },
    "Ecuador": {
        "pre_start":  "2008-04-15", "pre_end":  "2008-05-01",
        "post_start": "2008-05-01", "post_end": "2008-05-20",
        "bbox": [-77.5, -1.5, -76.0, -0.5],
        "split": "val"
    },
    "Ghana": {
        "pre_start":  "2019-07-15", "pre_end":  "2019-08-01",
        "post_start": "2019-08-01", "post_end": "2019-08-20",
        "bbox": [-1.5, 9.5, 0.0, 10.5],
        "split": "train"
    },
    "India": {
        "pre_start":  "2017-07-15", "pre_end":  "2017-08-01",
        "post_start": "2017-08-01", "post_end": "2017-08-20",
        "bbox": [75.5, 18.5, 77.0, 20.0],
        "split": "train"
    },
    "Mekong": {
        "pre_start":  "2019-07-15", "pre_end":  "2019-08-01",
        "post_start": "2019-08-01", "post_end": "2019-08-20",
        "bbox": [103.0, 14.5, 104.5, 16.0],
        "split": "train"
    },
    "Nigeria": {
        "pre_start":  "2018-09-15", "pre_end":  "2018-10-01",
        "post_start": "2018-10-01", "post_end": "2018-10-20",
        "bbox": [7.5, 6.5, 9.0, 8.0],
        "split": "train"
    },
    "Paraguay": {
        "pre_start":  "2019-01-15", "pre_end":  "2019-02-01",
        "post_start": "2019-02-01", "post_end": "2019-02-20",
        "bbox": [-59.5, -22.0, -58.0, -20.5],
        "split": "val"
    },
    "Somalia": {
        "pre_start":  "2019-10-15", "pre_end":  "2019-11-01",
        "post_start": "2019-11-01", "post_end": "2019-11-20",
        "bbox": [43.5, 1.5, 45.0, 3.0],
        "split": "train"
    },
}


def export_stats_csv(output_dir: str = "data"):
    """
    Computes HAND and WorldPop statistics per event region.
    No GEE export tasks — runs entirely client-side in ~3 minutes.
    Saves to data/event_stats.csv
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "event_stats.csv"
    rows = []

    for event_name, event in FLOOD_EVENTS.items():
        print(f"  {event_name}...", end=" ", flush=True)
        west, south, east, north = event["bbox"]
        aoi = ee.Geometry.Rectangle([west, south, east, north])

        hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
        pop  = (ee.ImageCollection("WorldPop/GP/100m/pop")
                .filter(ee.Filter.eq("year", 2020))
                .mosaic())

        hand_stats = hand.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=aoi, scale=90, maxPixels=1e8
        ).getInfo()

        pop_stats = pop.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=aoi, scale=100, maxPixels=1e8
        ).getInfo()

        row = {
            "event":    event_name,
            "split":    event["split"],
            "hand_mean": round(hand_stats.get("hnd_mean",   0) or 0, 3),
            "hand_std":  round(hand_stats.get("hnd_stdDev", 5) or 5, 3),
            "pop_mean":  round(pop_stats.get("population_mean",  0) or 0, 3),
            "pop_std":   round(pop_stats.get("population_stdDev",1) or 1, 3),
        }
        rows.append(row)
        print(f"HAND={row['hand_mean']:.1f}m  pop={row['pop_mean']:.1f}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → {out_path}")
    return str(out_path)


def export_hand_rasters(drive_folder: str, split: str = None):
    """
    Exports HAND + WorldPop as GeoTIFF at 30m resolution.
    Tight bounding boxes (~1.5x1.5 degree) = ~5,000x5,000 pixels.
    Each task completes in ~5-10 minutes. 
    """
    tasks = []
    for event_name, event in FLOOD_EVENTS.items():
        if split and event["split"] != split:
            continue

        west, south, east, north = event["bbox"]
        aoi  = ee.Geometry.Rectangle([west, south, east, north])

        hand = (ee.Image("MERIT/Hydro/v1_0_1")
                .select("hnd").rename("HAND")
                .min(100).max(0).clip(aoi))

        pop  = (ee.ImageCollection("WorldPop/GP/100m/pop")
                .filter(ee.Filter.eq("year", 2020))
                .mosaic().rename("population").clip(aoi))

        stack     = hand.addBands(pop).toFloat()
        task_name = f"hand_pop_{event_name}_{event['split']}"

        task = ee.batch.Export.image.toDrive(
            image=stack, description=task_name,
            folder=drive_folder, fileNamePrefix=task_name,
            region=aoi, scale=30, crs="EPSG:4326",
            maxPixels=1e9, fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append(task)
        print(f"  Launched: {task_name}")

    return tasks


def monitor_tasks(tasks, poll_interval=30):
    if not tasks:
        return
    print(f"\nMonitoring {len(tasks)} tasks...")
    pending = {t.id: t for t in tasks}
    while pending:
        time.sleep(poll_interval)
        still = {}
        for tid, task in pending.items():
            state = task.status()["state"]
            name  = task.status().get("description", tid)
            if state == "COMPLETED":
                print(f"  ✓ {name}")
            elif state in ("FAILED", "CANCELLED"):
                print(f"  ✗ {name} — {state}")
            else:
                still[tid] = task
        pending = still
        if pending:
            print(f"  ... {len(pending)} still running")
    print("All tasks complete.")


def main(args):
    ee.Authenticate()
    ee.Initialize(project=args.project)
    print(f"GEE initialised — project: {args.project}\n")

    tasks = []

    if args.mode in ("stats", "all"):
        print("=== Exporting statistics CSV (no GEE tasks, ~3 min) ===")
        export_stats_csv(output_dir=args.output_dir)

    if args.mode in ("rasters", "all"):
        print("\n=== Exporting HAND rasters at 30m ===")
        tasks = export_hand_rasters(
            drive_folder=args.drive_folder,
            split=args.split,
        )
        Path("task_ids.json").write_text(
            json.dumps({t.id: t.status()["description"] for t in tasks}, indent=2)
        )
        print(f"\nTask IDs saved → task_ids.json")
        print(f"Monitor: https://code.earthengine.google.com/tasks")
        if args.monitor:
            monitor_tasks(tasks)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",      type=str, required=True)
    parser.add_argument("--drive_folder", type=str, default="flood_hand_30m")
    parser.add_argument("--output_dir",   type=str, default="data")
    parser.add_argument("--mode",         type=str, default="all",
                        choices=["stats", "rasters", "all"])
    parser.add_argument("--split",        type=str, default=None)
    parser.add_argument("--monitor",      action="store_true")
    args = parser.parse_args()
    main(args)