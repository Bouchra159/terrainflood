"""
GEE Export Watchdog
====================
Monitors all active GEE export tasks, auto-retries failures up to MAX_RETRIES,
and writes a full log to export_watchdog.log.

Usage:
    python export_watchdog.py --project fast-kiln-489119-e1 \
                              --drive_folder flood_chips_train \
                              --split train

Leave running overnight. Check export_watchdog.log in the morning.
"""

import ee
import time
import json
import logging
import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL = 60          # seconds between status checks
MAX_RETRIES   = 3           # how many times to re-submit a failed task
LOG_FILE      = "export_watchdog.log"


# ── Logging: both console and file ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("watchdog")


# ── Load 01_gee_export functions without triggering argparse ──────────────────
def load_export_module():
    spec = importlib.util.spec_from_file_location("gee_export", "01_gee_export.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["gee_export"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Task state helpers ────────────────────────────────────────────────────────
TERMINAL_OK  = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}
RUNNING      = {"RUNNING", "READY", "UNSUBMITTED"}


def poll_tasks(task_objs: dict) -> dict:
    """Returns {task_id: status_dict} for all tracked tasks."""
    statuses = {}
    for tid, t in task_objs.items():
        try:
            statuses[tid] = t.status()
        except Exception as ex:
            statuses[tid] = {"state": "UNKNOWN", "description": str(ex)}
    return statuses


# ── Main watchdog loop ────────────────────────────────────────────────────────

def run_watchdog(args):
    log.info("=" * 60)
    log.info("GEE Export Watchdog started")
    log.info(f"  project      : {args.project}")
    log.info(f"  drive_folder : {args.drive_folder}")
    log.info(f"  split        : {args.split}")
    log.info(f"  max_retries  : {MAX_RETRIES}")
    log.info(f"  poll_interval: {POLL_INTERVAL}s")
    log.info("=" * 60)

    ee.Initialize(project=args.project)
    mod = load_export_module()

    # Submit the initial batch of tasks
    log.info("Submitting initial export tasks...")
    tasks        = {}   # task_id -> ee.batch.Task object
    retry_count  = {}   # event_name -> int
    event_map    = {}   # task_id -> event_name  (for retry)

    for event_name, event in mod.FLOOD_EVENTS.items():
        if event["split"] != args.split:
            continue
        try:
            stack, aoi = mod.build_stack(event_name, event)
            task_name  = f"flood_{event_name}_{args.split}"
            t = mod.launch_export_tfrecord(
                image        = stack,
                aoi          = aoi,
                task_name    = task_name,
                drive_folder = args.drive_folder,
            )
            tasks[t.id]       = t
            retry_count[event_name] = 0
            event_map[t.id]   = event_name
            log.info(f"  Submitted: {event_name}")
        except Exception as ex:
            log.error(f"  FAILED to submit {event_name}: {ex}")

    if not tasks:
        log.error("No tasks were submitted. Check errors above. Exiting.")
        return

    log.info(f"Monitoring {len(tasks)} tasks. Will check every {POLL_INTERVAL}s.\n")

    completed = set()
    failed    = set()

    # ── Poll loop ─────────────────────────────────────────────────────────────
    while True:
        time.sleep(POLL_INTERVAL)

        still_running = {}
        statuses = poll_tasks(tasks)

        for tid, status in statuses.items():
            state      = status.get("state", "UNKNOWN")
            name       = status.get("description", tid)
            event_name = event_map.get(tid, "unknown")

            if state in TERMINAL_OK:
                if tid not in completed:
                    log.info(f"  COMPLETED  {name}")
                    completed.add(tid)

            elif state in TERMINAL_BAD:
                if tid not in failed:
                    err_msg = status.get("error_message", "no details")
                    log.warning(f"  FAILED     {name}  ({state}): {err_msg}")
                    failed.add(tid)

                    # Auto-retry if under limit
                    retries = retry_count.get(event_name, 0)
                    if retries < MAX_RETRIES:
                        log.info(f"  RETRYING   {event_name} "
                                 f"(attempt {retries + 1}/{MAX_RETRIES})...")
                        try:
                            stack, aoi = mod.build_stack(
                                event_name, mod.FLOOD_EVENTS[event_name]
                            )
                            task_name = f"flood_{event_name}_{args.split}"
                            new_t = mod.launch_export_tfrecord(
                                image        = stack,
                                aoi          = aoi,
                                task_name    = task_name,
                                drive_folder = args.drive_folder,
                            )
                            tasks[new_t.id]          = new_t
                            event_map[new_t.id]      = event_name
                            retry_count[event_name]  = retries + 1
                            still_running[new_t.id]  = new_t
                            log.info(f"  Re-submitted {event_name} → task {new_t.id}")
                        except Exception as ex:
                            log.error(f"  Re-submit FAILED for {event_name}: {ex}")
                    else:
                        log.error(f"  GIVING UP  {event_name} — "
                                  f"exceeded {MAX_RETRIES} retries")

            else:
                # Still running
                still_running[tid] = tasks[tid]

        tasks = still_running

        n_done    = len(completed)
        n_failed  = len([e for e, r in retry_count.items()
                         if r >= MAX_RETRIES and
                         not any(event_map.get(tid) == e for tid in completed)])
        n_running = len(tasks)

        log.info(f"  Status: {n_done} done | {n_running} running | "
                 f"{n_failed} gave up")

        if not tasks:
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("WATCHDOG COMPLETE — Morning summary")
    log.info(f"  Completed : {len(completed)} tasks")
    log.info(f"  Total time: see timestamps in {LOG_FILE}")

    gave_up = [e for e, r in retry_count.items() if r >= MAX_RETRIES]
    if gave_up:
        log.warning(f"  NEEDS ATTENTION: {', '.join(gave_up)}")
        log.warning("  These events failed all retries. Check GEE quota or data.")
    else:
        log.info("  All events exported successfully.")
    log.info("=" * 60)

    # Save final task IDs
    Path("task_ids.json").write_text(
        json.dumps({tid: event_map.get(tid, "?") for tid in completed}, indent=2)
    )
    log.info(f"Completed task IDs saved → task_ids.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEE Export Watchdog with auto-retry")
    parser.add_argument("--project",      required=True)
    parser.add_argument("--drive_folder", required=True)
    parser.add_argument("--split",        required=True,
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    try:
        run_watchdog(args)
    except KeyboardInterrupt:
        log.info("Watchdog interrupted by user (Ctrl+C).")
    except Exception as ex:
        log.exception(f"Watchdog crashed: {ex}")
        log.error("Check export_watchdog.log for details.")
