"""
tools/audit_pipeline.py
========================
Pre-training pipeline audit for TerrainFlood-UQ.

Run this BEFORE any training run to verify that:
  A) Dataset discovery is correct and paths exist
  B) Split mapping is correct (Bolivia only in test, no overlap)
  C) Label value distribution is sane (flood ratio, ignore pixels)
  D) IoU function is correct (unit test with synthetic data)
  E) Validation loop iterates ALL batches
  F) WeightedRandomSampler produces batches with flood pixels

Usage:
  cd ~/terrainflood
  python tools/audit_pipeline.py --data_root data/sen1floods11

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (see FAIL lines in output).
"""

import argparse
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# ── Module imports (handles numeric-prefixed filenames) ──────────────────────

_repo_root = Path(__file__).parent.parent


def _import_module(alias: str, file_path: str):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "dataset" not in sys.modules:
    _import_module("dataset", str(_repo_root / "02_dataset.py"))
if "model" not in sys.modules:
    _import_module("model",   str(_repo_root / "03_model.py"))

from dataset import (  # noqa: E402
    FloodDataset,
    get_dataloaders,
    SPLIT_MAP,
    LABEL_FLOOD,
    LABEL_IGNORE,
    LABEL_PERMANENT_WATER,
)
from model import build_model, FloodLoss  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_WARN = "\033[33mWARN\033[0m"
_failures: list[str] = []


def _ok(msg: str) -> None:
    print(f"  [{_PASS}] {msg}")


def _fail(msg: str) -> None:
    print(f"  [{_FAIL}] {msg}")
    _failures.append(msg)


def _warn(msg: str) -> None:
    print(f"  [{_WARN}] {msg}")


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# A.  Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def audit_discovery(data_root: str) -> dict[str, int]:
    """Loads all three splits and prints sample counts + example paths."""
    _section("A. Dataset Discovery")
    counts: dict[str, int] = {}

    for split in ["train", "val", "test"]:
        try:
            ds = FloodDataset(data_root, split=split, patch_size=None,
                              augment=False, normalize=False)
            counts[split] = len(ds)
            _ok(f"{split:5s}  {len(ds):4d} chips")

            # Print first 3 examples
            for i in range(min(3, len(ds))):
                s = ds.samples[i]
                label_exists = Path(s["label_path"]).exists()
                s1_exists    = Path(s["s1_path"]).exists()
                status = "OK" if (label_exists and s1_exists) else "MISSING"
                print(f"         {status}  event={s['event']:<15}  "
                      f"chip={s['chip_id']}")
                if not label_exists:
                    _fail(f"Label file missing: {s['label_path']}")
                if not s1_exists:
                    _fail(f"S1 file missing: {s['s1_path']}")

        except Exception as exc:
            _fail(f"split={split} failed to load: {exc}")
            counts[split] = 0

    total = sum(counts.values())
    if total == 0:
        _fail("No chips found across any split — check data_root path")
    elif total < 50:
        _warn(f"Very few chips total ({total}); dataset may be incomplete")
    else:
        _ok(f"Total chips: {total}")

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# B.  Split mapping correctness
# ─────────────────────────────────────────────────────────────────────────────

def audit_splits(data_root: str) -> None:
    """Verifies Bolivia only in test, no cross-split overlap, known events."""
    _section("B. Split Mapping")

    # Discover all events present in the actual label files
    label_dir = Path(data_root) / "flood_events" / "HandLabeled" / "LabelHand"
    if not label_dir.exists():
        _fail(f"Label dir not found: {label_dir}")
        return

    found_events = Counter(
        p.name.split("_")[0]
        for p in label_dir.glob("*_LabelHand.tif")
    )
    print(f"\n  Events in dataset: {dict(found_events.most_common())}")
    print(f"  Events in SPLIT_MAP: {sorted(SPLIT_MAP.keys())}\n")

    # Check for events present on disk but NOT in SPLIT_MAP
    unmapped = [e for e in found_events if e not in SPLIT_MAP]
    if unmapped:
        _warn(f"Events on disk but not in SPLIT_MAP (default→train): {unmapped}")
    else:
        _ok("All disk events are in SPLIT_MAP")

    # Load all three splits
    datasets: dict[str, FloodDataset] = {}
    for split in ["train", "val", "test"]:
        try:
            datasets[split] = FloodDataset(
                data_root, split=split, patch_size=None, augment=False
            )
        except Exception as exc:
            _fail(f"Could not load split={split}: {exc}")
            return

    tr, va, te = datasets["train"], datasets["val"], datasets["test"]

    # Chip-level overlap
    tr_ids = {s["chip_id"] for s in tr.samples}
    va_ids = {s["chip_id"] for s in va.samples}
    te_ids = {s["chip_id"] for s in te.samples}

    tv_overlap = tr_ids & va_ids
    tt_overlap = tr_ids & te_ids
    vt_overlap = va_ids & te_ids

    if tv_overlap:
        _fail(f"train∩val overlap: {len(tv_overlap)} chips — DATA LEAKAGE")
    else:
        _ok("train ∩ val = ∅  (no leakage)")

    if tt_overlap:
        _fail(f"train∩test overlap: {len(tt_overlap)} chips — DATA LEAKAGE")
    else:
        _ok("train ∩ test = ∅  (no leakage)")

    if vt_overlap:
        _fail(f"val∩test overlap: {len(vt_overlap)} chips — DATA LEAKAGE")
    else:
        _ok("val ∩ test = ∅  (no leakage)")

    # Bolivia rule
    bolivia_in_train = any(s["event"] == "Bolivia" for s in tr.samples)
    bolivia_in_val   = any(s["event"] == "Bolivia" for s in va.samples)
    bolivia_in_test  = any(s["event"] == "Bolivia" for s in te.samples)

    if bolivia_in_train:
        _fail("Bolivia found in TRAIN split — violates OOD evaluation protocol")
    else:
        _ok("Bolivia NOT in train")

    if bolivia_in_val:
        _fail("Bolivia found in VAL split — violates OOD evaluation protocol")
    else:
        _ok("Bolivia NOT in val")

    if not bolivia_in_test:
        _warn("Bolivia NOT in test split — is data downloaded?")
    else:
        n_bolivia = sum(1 for s in te.samples if s["event"] == "Bolivia")
        _ok(f"Bolivia in test: {n_bolivia} chips")

    # Print per-split event breakdown
    for split, ds in datasets.items():
        ec = Counter(s["event"] for s in ds.samples)
        print(f"\n  {split} ({len(ds.samples)} chips): {dict(ec.most_common())}")


# ─────────────────────────────────────────────────────────────────────────────
# C.  Label value distribution
# ─────────────────────────────────────────────────────────────────────────────

def audit_labels(data_root: str, n_sample: int = 100) -> None:
    """Checks label value distribution on a sample of training chips."""
    _section("C. Label Distribution")

    try:
        ds = FloodDataset(data_root, split="train", patch_size=None,
                          augment=False, normalize=False)
    except Exception as exc:
        _fail(f"Cannot load train split: {exc}")
        return

    n = min(n_sample, len(ds))
    total_px = flood_px = ignore_px = water_px = 0
    chips_with_flood = 0

    for i in range(n):
        sample = ds[i]
        lbl    = sample["label"].numpy()   # (H, W)  int64

        valid_mask = lbl != LABEL_IGNORE
        total_px  += int(valid_mask.sum())
        flood_px  += int((lbl == LABEL_FLOOD).sum())
        ignore_px += int((lbl == LABEL_IGNORE).sum())
        water_px  += int((lbl == LABEL_PERMANENT_WATER).sum())

        if (lbl == LABEL_FLOOD).any():
            chips_with_flood += 1

    flood_ratio = flood_px / max(total_px, 1)
    ignore_ratio = ignore_px / max(total_px + ignore_px, 1)

    print(f"\n  Sample: {n} chips from train split")
    print(f"  Total valid pixels    : {total_px:,}")
    print(f"  Flood pixels (label=1): {flood_px:,}  ({flood_ratio*100:.2f}%)")
    print(f"  Ignore pixels (=-1)   : {ignore_px:,}  ({ignore_ratio*100:.2f}%)")
    print(f"  Perm. water (label=2) : {water_px:,}")
    print(f"  Chips with any flood  : {chips_with_flood}/{n}")

    if flood_ratio < 0.001:
        _fail(f"Flood pixel ratio is extremely low ({flood_ratio:.4f}) — "
              "labels may be all-zero or label reading is broken")
    elif flood_ratio < 0.01:
        _warn(f"Low flood ratio ({flood_ratio:.4f}) — class imbalance is severe; "
              "verify pos_weight and sampler are active")
    else:
        _ok(f"Flood pixel ratio = {flood_ratio:.4f}  "
            f"({chips_with_flood}/{n} chips contain flood)")

    if ignore_ratio > 0.5:
        _warn(f"More than 50% pixels are ignore (-1); check nodata handling")
    else:
        _ok(f"Ignore pixel ratio = {ignore_ratio:.4f}  (reasonable)")

    # Check unique label values to catch unexpected values
    print(f"\n  Checking unique label values (first {n} chips)...")
    all_unique: set[int] = set()
    for i in range(n):
        lbl = ds[i]["label"].numpy()
        all_unique.update(np.unique(lbl).tolist())

    expected = {-1, 0, 1}
    unexpected = all_unique - expected - {2}  # 2 is optional (permanent water)
    if unexpected:
        _fail(f"Unexpected label values found: {unexpected} — "
              "nodata handling may be broken (check src.nodata)")
    else:
        _ok(f"Label values present: {sorted(all_unique)}")


# ─────────────────────────────────────────────────────────────────────────────
# D.  IoU function unit test
# ─────────────────────────────────────────────────────────────────────────────

def audit_iou() -> None:
    """Unit tests for the compute_iou function in train.py."""
    _section("D. IoU Function Unit Tests")

    # Import compute_iou from train.py
    train_mod = _import_module("train_mod", str(_repo_root / "train.py"))
    compute_iou = train_mod.compute_iou

    device = torch.device("cpu")

    # Test 1: perfect prediction
    logits  = torch.tensor([[[[5.0, 5.0], [-5.0, -5.0]]]])   # (1,1,2,2)
    labels  = torch.tensor([[[1, 1, 0, 0]]]).reshape(1, 2, 2)  # (1,2,2)
    iou = compute_iou(logits, labels)
    if abs(iou - 1.0) < 1e-4:
        _ok(f"Perfect prediction IoU = {iou:.4f}  (expected 1.0)")
    else:
        _fail(f"Perfect prediction IoU = {iou:.4f}  (expected 1.0)")

    # Test 2: all-zero prediction (model predicts no flood)
    logits_zero = torch.full((1, 1, 4, 4), -5.0)
    labels_pos  = torch.ones(1, 4, 4).long()
    iou_zero = compute_iou(logits_zero, labels_pos)
    if abs(iou_zero) < 1e-4:
        _ok(f"All-zero prediction IoU = {iou_zero:.4f}  (expected 0.0)")
    else:
        _fail(f"All-zero prediction IoU = {iou_zero:.4f}  (expected ~0.0)")

    # Test 3: ignore pixels should not count
    logits_mix = torch.tensor([[[[5.0, 5.0, 5.0, 5.0]]]])     # predicts all flood
    labels_mix = torch.tensor([[[1, 0, -1, -1]]])              # 1 flood, 1 non-flood, 2 ignore
    iou_mix = compute_iou(logits_mix.reshape(1, 1, 1, 4),
                          labels_mix.reshape(1, 1, 4))
    # TP=1, FP=1, FN=0 → IoU = 1/(1+1+0) = 0.5
    expected_iou = 1.0 / (1.0 + 1.0 + 0.0 + 1e-6)
    if abs(iou_mix - expected_iou) < 1e-3:
        _ok(f"Ignore mask IoU = {iou_mix:.4f}  (expected {expected_iou:.4f})")
    else:
        _fail(f"Ignore mask IoU = {iou_mix:.4f}  (expected {expected_iou:.4f}) — "
              "ignore pixels not handled correctly")

    # Test 4: all-ignore labels → should return ~0 gracefully
    labels_all_ignore = torch.full((1, 4, 4), -1).long()
    logits_any = torch.randn(1, 1, 4, 4)
    iou_all_ignore = compute_iou(logits_any, labels_all_ignore)
    if iou_all_ignore == 0.0:
        _ok(f"All-ignore labels IoU = {iou_all_ignore:.4f}  (no crash, returns 0)")
    else:
        _warn(f"All-ignore labels IoU = {iou_all_ignore:.4f}  (expected 0)")


# ─────────────────────────────────────────────────────────────────────────────
# E.  Validation loop completeness
# ─────────────────────────────────────────────────────────────────────────────

def audit_val_loop(data_root: str) -> None:
    """
    Verifies the val DataLoader iterates ALL batches and computes metrics.
    Uses a randomly initialised model (no checkpoint required).
    """
    _section("E. Validation Loop")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    try:
        _, val_loader, _ = get_dataloaders(
            data_root   = data_root,
            batch_size  = 1,
            num_workers = 0,
            patch_size  = 256,
            pin_memory  = False,
        )
    except Exception as exc:
        _fail(f"Could not create val_loader: {exc}")
        return

    n_val_chips = len(val_loader.dataset)
    print(f"  Val dataset size : {n_val_chips} chips")
    print(f"  Val loader size  : {len(val_loader)} batches")

    if n_val_chips == 0:
        _fail("Validation set is empty!")
        return

    # Quick model
    model = build_model("A", pretrained=False).to(device)
    model.eval()
    criterion = FloodLoss(pos_weight=10.0, alpha=0.5)

    train_mod = _import_module("train_mod_val", str(_repo_root / "train.py"))
    val_epoch = train_mod.val_epoch

    print(f"\n  Running full val epoch (random model)...")
    t0 = time.time()
    metrics, first_batch = val_epoch(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    print(f"  Val epoch time   : {elapsed:.1f}s")
    print(f"  Val loss         : {metrics['loss']:.4f}")
    print(f"  Val IoU          : {metrics['iou']:.4f}  "
          f"(random model → expect near 0)")
    print(f"  First batch keys : {list(first_batch.keys()) if first_batch else 'None'}")

    if first_batch is None:
        _fail("val_epoch returned first_batch=None — val loader may be empty")
    else:
        _ok("Val loop completed and returned first_batch")

    # Check batch shapes
    if first_batch is not None:
        img_shape = first_batch["image"].shape
        lbl_shape = first_batch["label"].shape
        if img_shape[1] == 6:
            _ok(f"Batch image shape: {tuple(img_shape)} — 6 bands correct")
        else:
            _fail(f"Batch image shape: {tuple(img_shape)} — expected 6 bands")
        _ok(f"Batch label shape: {tuple(lbl_shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# F.  WeightedRandomSampler — flood content in training batches
# ─────────────────────────────────────────────────────────────────────────────

def audit_sampler(data_root: str, n_batches: int = 20) -> None:
    """
    Draws n_batches from the train loader and measures flood pixel fraction.
    If the sampler is working, batches should have a higher flood ratio than
    the raw dataset.
    """
    _section("F. WeightedRandomSampler — Training Batch Flood Content")

    try:
        train_loader, _, _ = get_dataloaders(
            data_root   = data_root,
            batch_size  = 4,
            num_workers = 0,
            patch_size  = 256,
            pin_memory  = False,
        )
    except Exception as exc:
        _fail(f"Could not create train_loader: {exc}")
        return

    print(f"  Sampling {n_batches} training batches...")

    batch_flood_fracs = []
    batches_with_flood = 0

    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        labels = batch["label"].numpy()   # (B, H, W)
        valid  = labels != LABEL_IGNORE
        n_valid = valid.sum()
        n_flood = (labels == LABEL_FLOOD).sum()
        frac = float(n_flood) / max(int(n_valid), 1)
        batch_flood_fracs.append(frac)
        if (labels == LABEL_FLOOD).any():
            batches_with_flood += 1

    mean_frac = float(np.mean(batch_flood_fracs))
    min_frac  = float(np.min(batch_flood_fracs))
    max_frac  = float(np.max(batch_flood_fracs))

    print(f"  Batches sampled         : {len(batch_flood_fracs)}")
    print(f"  Batches with flood      : {batches_with_flood}/{len(batch_flood_fracs)}")
    print(f"  Mean flood pixel frac   : {mean_frac:.4f}")
    print(f"  Range                   : [{min_frac:.4f}, {max_frac:.4f}]")

    if batches_with_flood == 0:
        _fail("No training batches contain any flood pixels! "
              "Model cannot learn to detect floods.")
    elif batches_with_flood < len(batch_flood_fracs) * 0.5:
        _warn(f"Only {batches_with_flood}/{len(batch_flood_fracs)} batches contain flood — "
              "sampler may not be effective enough")
    else:
        _ok(f"{batches_with_flood}/{len(batch_flood_fracs)} batches contain flood pixels")

    if mean_frac < 0.005:
        _warn(f"Mean flood fraction very low ({mean_frac:.4f}); "
              "consider higher pos_weight or stronger sampler")
    else:
        _ok(f"Mean batch flood fraction = {mean_frac:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TerrainFlood-UQ Pipeline Audit")
    p.add_argument("--data_root", type=str, default="data/sen1floods11",
                   help="Root of Sen1Floods11 dataset")
    p.add_argument("--n_label_sample", type=int, default=100,
                   help="Number of chips to scan for label distribution (check C)")
    p.add_argument("--n_sampler_batches", type=int, default=20,
                   help="Number of training batches to draw for sampler check (check F)")
    p.add_argument("--skip_val_loop", action="store_true",
                   help="Skip check E (val loop) — faster but less thorough")
    p.add_argument("--skip_sampler", action="store_true",
                   help="Skip check F (sampler) — faster")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  TerrainFlood-UQ  —  Pipeline Audit")
    print(f"  data_root = {args.data_root}")
    print("=" * 60)

    audit_discovery(args.data_root)
    audit_splits(args.data_root)
    audit_labels(args.data_root, n_sample=args.n_label_sample)
    audit_iou()

    if not args.skip_val_loop:
        audit_val_loop(args.data_root)
    else:
        print("\n  [Skipped] E. Validation loop (--skip_val_loop)")

    if not args.skip_sampler:
        audit_sampler(args.data_root, n_batches=args.n_sampler_batches)
    else:
        print("\n  [Skipped] F. Sampler check (--skip_sampler)")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    if _failures:
        print(f"  AUDIT FAILED  —  {len(_failures)} issue(s) found:\n")
        for i, msg in enumerate(_failures, 1):
            print(f"    {i}. {msg}")
        print(f"\n{'='*60}")
        sys.exit(1)
    else:
        print(f"  AUDIT PASSED  —  all checks OK")
        print(f"{'='*60}")
        sys.exit(0)


if __name__ == "__main__":
    main()
