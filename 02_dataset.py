"""
Phase 1 — Dataset Loader
=========================
Handles Sen1Floods11 local chips (downloaded via gsutil).

IMPORTANT (your current DKUCC layout):
data_root/
  flood_events/
    HandLabeled/
      S1Hand/        *_S1Hand.tif        (2-band VV,VH)
      LabelHand/     *_LabelHand.tif     (1-band labels: 0,1,2)
      S2Hand/        *_S2Hand.tif        (optional, not used)
      ... (other hand-labeled dirs)
    WeaklyLabeled/   (ignored by this loader)

Optional (only if you have per-chip exports; otherwise zeros are used):
  hand_chips/        *_HAND.tif
  pop_chips/         *_pop.tif

This loader returns dict batches:
  {
    "image":    (6,H,W) float32  [VV_pre, VH_pre, VV_post, VH_post, ratio, HAND]
    "label":    (H,W)   int64
    "hand_raw": (H,W)   float32
    "pop":      (H,W)   float32   # NOT model input, used only in exposure step
    "chip_id":  str
    "event":    str
  }
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import rasterio
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Band statistics for normalisation
# (These are placeholder-ish; update after first run if you want.)
# 6-band model input: [VV_pre, VH_pre, VV_post, VH_post, VV_VH_ratio, HAND]
# WorldPop is NOT included — it is returned separately as batch["pop"].
# ─────────────────────────────────────────────
NORM_STATS = {
    "VV_pre":       (-12.5,  4.5),
    "VH_pre":       (-19.5,  4.5),
    "VV_post":      (-12.5,  4.5),
    "VH_post":      (-19.5,  4.5),
    "VV_VH_ratio":  ( -7.0,  3.0),
    "HAND":         (  5.0, 10.0),
}

# Sen1Floods11 HandLabeled split — Bolivia is ALWAYS test (out-of-distribution)
# Full 11-event HandLabeled benchmark (446 chips total):
#   Train : 364 chips — 9 events
#   Val   :  67 chips — Paraguay only (standard benchmark split)
#   Test  :  15 chips — Bolivia OOD holdout
# Note: Cambodia, Canada, DemRepCongo, Ecuador are in the original SPLIT_MAP
# but have no HandLabeled chips in the downloaded dataset (kept for completeness).
SPLIT_MAP = {
    # ── Test (OOD holdout — never touch) ─────────────────────
    "Bolivia":     "test",
    # ── Validation ────────────────────────────────────────────
    "Ecuador":     "val",      # 0 chips in HandLabeled subset
    "Paraguay":    "val",      # 67 chips
    # ── Training ──────────────────────────────────────────────
    "Cambodia":    "train",    # 0 chips in HandLabeled subset
    "Canada":      "train",    # 0 chips in HandLabeled subset
    "DemRepCongo": "train",    # 0 chips in HandLabeled subset
    "Ghana":       "train",    # 53 chips
    "India":       "train",    # 68 chips
    "Mekong":      "train",    # 30 chips
    "Nigeria":     "train",    # 18 chips
    "Pakistan":    "train",    # 28 chips
    "Somalia":     "train",    # 26 chips
    "Spain":       "train",    # 30 chips
    "Sri-Lanka":   "train",    # 42 chips
    "USA":         "train",    # 69 chips
}

# Sen1Floods11 label values
LABEL_NO_FLOOD        = 0
LABEL_FLOOD           = 1
LABEL_PERMANENT_WATER = 2
LABEL_IGNORE          = -1   # masked / invalid pixels


class FloodDataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 (HandLabeled) + optional per-chip HAND/Pop.

    If hand_chips/ and pop_chips/ do not exist, HAND and population
    are set to zeros (graceful degradation).

    Supported splits:
      "train"     — 9 events, HandLabeled, 364 chips
      "val"       — Paraguay only, HandLabeled, 67 chips
      "test"      — Bolivia only, HandLabeled, 15 chips (OOD holdout)
      "test_weak" — All events in WeaklyLabeled (noisy labels, Bolivia excluded).
                    Useful for assessing model generalisation on weakly-supervised data.
                    Falls back gracefully if WeaklyLabeled directory does not exist.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",          # "train" | "val" | "test" | "test_weak"
        patch_size: int = 256,         # crop size during training
        augment: bool = True,
        normalize: bool = True,
        include_hand: bool = True,
        include_pop: bool = True,
        permanent_water: str = "include",  # "include" | "exclude" | "flood"
    ):
        self.data_root       = Path(data_root)
        self.split           = split
        self.patch_size      = patch_size
        self.augment         = augment and (split == "train")
        self.normalize       = normalize
        self.include_hand    = include_hand
        self.include_pop     = include_pop
        self.permanent_water = permanent_water

        self.samples = self._discover_samples()
        self.class_weights = self._compute_class_weights()

        # Auto-load per-band statistics from norm_stats.json if present.
        # This file is written by compute_normalization_stats() below and
        # contains the real HAND mean/std computed on actual HAND chips.
        # Falls back to the module-level NORM_STATS placeholders otherwise.
        self._norm_stats = NORM_STATS.copy()
        norm_json = self.data_root / "norm_stats.json"
        if norm_json.exists():
            with open(norm_json) as _f:
                _loaded = json.load(_f)
            self._norm_stats.update({k: tuple(v) for k, v in _loaded.items()})
            print(f"[FloodDataset] Loaded norm stats from {norm_json}")
        else:
            print(
                "[FloodDataset] WARNING: using placeholder HAND norm stats "
                "(mean=5, std=10). Run:  python 02_dataset.py <data_root>  "
                "after HAND chips are ready to compute real statistics."
            )

        print(
            f"[FloodDataset] split={split} chips={len(self.samples)} "
            f"flood_ratio={self.class_weights['flood_ratio']:.3f} "
            f"pos_weight={self.class_weights['pos_weight']:.2f}"
        )

    # ─── sample discovery ───────────────────

    def _discover_samples(self) -> list[dict]:
        """
        Finds Sen1Floods11 chips for the requested split.

        HandLabeled splits (train/val/test):
          Supports BOTH layouts:
            A) Recommended: flood_events/HandLabeled/LabelHand/*_LabelHand.tif
               +           flood_events/HandLabeled/S1Hand/*_S1Hand.tif
            B) Flat:        flood_events/*_LabelHand.tif

        WeaklyLabeled split (test_weak):
          flood_events/WeaklyLabeled/S1Weak/*_S1Weak.tif
          flood_events/WeaklyLabeled/LabelWeak/*_LabelWeak.tif
          Bolivia is excluded (OOD holdout). Falls back gracefully if not present.
        """
        if self.split == "test_weak":
            return self._discover_weak_samples()

        event_dir = self.data_root / "flood_events"
        if not event_dir.exists():
            raise FileNotFoundError(
                f"Expected Sen1Floods11 chips at {event_dir}\n"
                f"Download with: gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events {self.data_root}/"
            )

        # Prefer HandLabeled layout if present
        hand_root = event_dir / "HandLabeled"
        if hand_root.exists():
            label_dir = hand_root / "LabelHand"
            s1_dir    = hand_root / "S1Hand"
        else:
            label_dir = event_dir
            s1_dir    = event_dir

        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        if not s1_dir.exists():
            raise FileNotFoundError(f"S1 directory not found: {s1_dir}")

        samples: list[dict] = []

        for label_path in sorted(label_dir.glob("*_LabelHand.tif")):
            chip_id = label_path.stem.replace("_LabelHand", "")
            event_name = chip_id.split("_")[0]

            # Split filtering (Bolivia sacred as test)
            if SPLIT_MAP.get(event_name, "train") != self.split:
                continue

            s1_path = s1_dir / f"{chip_id}_S1Hand.tif"
            if not s1_path.exists():
                # Skip if missing the SAR for this label
                continue

            # Optional per-chip HAND and pop (only if you exported them per chip)
            hand_path = self.data_root / "hand_chips" / f"{chip_id}_HAND.tif"
            pop_path  = self.data_root / "pop_chips"  / f"{chip_id}_pop.tif"

            samples.append({
                "chip_id":    chip_id,
                "event":      event_name,
                "s1_path":    s1_path,
                "label_path": label_path,
                "hand_path":  hand_path if hand_path.exists() else None,
                "pop_path":   pop_path  if pop_path.exists()  else None,
            })

        if not samples and self.split not in ("test_weak",):
            raise RuntimeError(
                f"No chips found for split='{self.split}' in:\n"
                f"  labels: {label_dir}\n"
                f"  s1:     {s1_dir}\n"
                f"Check SPLIT_MAP and that HandLabeled data is downloaded."
            )

        return samples

    def _discover_weak_samples(self) -> list[dict]:
        """
        Finds WeaklyLabeled chips (split='test_weak').

        Bolivia is always excluded even from WeaklyLabeled.
        Returns an empty list (not an error) if the WeaklyLabeled directory
        does not exist — allows graceful degradation on setups without weak data.
        """
        event_dir = self.data_root / "flood_events"
        weak_root = event_dir / "WeaklyLabeled"

        if not weak_root.exists():
            print(
                f"[FloodDataset] WARNING: WeaklyLabeled directory not found at "
                f"{weak_root}. split='test_weak' returns 0 chips. "
                f"Download with: gsutil -m cp -r "
                f"gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled {event_dir}/"
            )
            return []

        label_dir = weak_root / "LabelWeak"
        s1_dir    = weak_root / "S1Weak"

        if not label_dir.exists() or not s1_dir.exists():
            print(
                f"[FloodDataset] WARNING: LabelWeak/ or S1Weak/ missing in "
                f"{weak_root}. split='test_weak' returns 0 chips."
            )
            return []

        samples: list[dict] = []

        for label_path in sorted(label_dir.glob("*_LabelWeak.tif")):
            chip_id    = label_path.stem.replace("_LabelWeak", "")
            event_name = chip_id.split("_")[0]

            # Always exclude Bolivia (OOD holdout — never contaminate test)
            if event_name == "Bolivia":
                continue

            s1_path = s1_dir / f"{chip_id}_S1Weak.tif"
            if not s1_path.exists():
                continue

            hand_path = self.data_root / "hand_chips" / f"{chip_id}_HAND.tif"
            pop_path  = self.data_root / "pop_chips"  / f"{chip_id}_pop.tif"

            samples.append({
                "chip_id":    chip_id,
                "event":      event_name,
                "s1_path":    s1_path,
                "label_path": label_path,
                "hand_path":  hand_path if hand_path.exists() else None,
                "pop_path":   pop_path  if pop_path.exists()  else None,
            })

        print(
            f"[FloodDataset] split=test_weak: found {len(samples)} WeaklyLabeled chips "
            f"(Bolivia excluded). Labels are noisy — use for generalisation analysis only."
        )
        return samples

    # ─── class weight estimation ─────────────

    def _compute_class_weights(self) -> dict:
        """
        Quick pass over label files to estimate flood pixel ratio.
        Used for WeightedRandomSampler and loss weighting.
        Returns safe defaults when samples list is empty (e.g. test_weak fallback).
        """
        if not self.samples:
            return {"flood_ratio": 0.05, "pos_weight": 10.0}
        total = 0
        flood = 0
        n = min(50, len(self.samples))
        for s in self.samples[:n]:
            try:
                with rasterio.open(s["label_path"]) as src:
                    label = src.read(1)
                valid = label[label != LABEL_IGNORE]
                flood += (valid == LABEL_FLOOD).sum()
                total += valid.size
            except Exception:
                pass

        ratio = float(flood) / max(total, 1)
        pos_weight = (1.0 - ratio) / max(ratio, 1e-6)
        return {"flood_ratio": ratio, "pos_weight": min(pos_weight, 20.0)}

    # ─── loading helpers ─────────────────────

    def _read_tif(self, path: Path, bands=None) -> np.ndarray:
        """Reads a GeoTIFF and returns a (C, H, W) float32 array."""
        with rasterio.open(path) as src:
            if bands is None:
                data = src.read().astype(np.float32)
            else:
                data = src.read(bands).astype(np.float32)
        return data

    def _normalise(self, band_data: np.ndarray, band_name: str) -> np.ndarray:
        """Zero-mean, unit-variance normalisation using per-instance stats.

        Uses self._norm_stats which is populated from norm_stats.json when
        available, otherwise falls back to module-level NORM_STATS defaults.
        """
        if band_name not in self._norm_stats:
            return band_data
        mean, std = self._norm_stats[band_name]
        return (band_data - mean) / (std + 1e-6)

    def _handle_permanent_water(self, label: np.ndarray) -> np.ndarray:
        """
        Permanent water pixels (class 2) handling:
          'include' → treat as flood (1)
          'exclude' → set to ignore (-1)
          'flood'   → keep as flood (1)
        """
        if self.permanent_water in ("include", "flood"):
            label = np.where(label == LABEL_PERMANENT_WATER, LABEL_FLOOD, label)
        elif self.permanent_water == "exclude":
            label = np.where(label == LABEL_PERMANENT_WATER, LABEL_IGNORE, label)
        return label

    def _random_crop(self, *arrays, size: int) -> tuple:
        """Applies the same random crop to all input arrays."""
        h, w = arrays[0].shape[-2], arrays[0].shape[-1]

        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            arrays = tuple(
                np.pad(a, [(0, 0), (0, pad_h), (0, pad_w)], mode="reflect")
                for a in arrays
            )
            h, w = arrays[0].shape[-2], arrays[0].shape[-1]

        top  = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
        return tuple(a[..., top:top + size, left:left + size] for a in arrays)

    def _augment(self, image: np.ndarray, label: np.ndarray) -> tuple:
        """Label-preserving augments: flips + 90° rotations."""
        if np.random.random() > 0.5:
            image = np.flip(image, axis=-1).copy()
            label = np.flip(label, axis=-1).copy()
        if np.random.random() > 0.5:
            image = np.flip(image, axis=-2).copy()
            label = np.flip(label, axis=-2).copy()
        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            label = np.rot90(label, k, axes=(-2, -1)).copy()
        return image, label

    # ─── __getitem__ ─────────────────────────

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # SAR: S1Hand has 2 bands (VV, VH)
        s1 = self._read_tif(Path(s["s1_path"]))  # (2, H, W)

        vv = s1[0:1]   # (1, H, W)
        vh = s1[1:2]   # (1, H, W)

        # Single-date baseline: pre = post = same chip
        if self.normalize:
            vv_pre  = self._normalise(vv.copy(), "VV_pre")
            vh_pre  = self._normalise(vh.copy(), "VH_pre")
            vv_post = self._normalise(vv.copy(), "VV_post")
            vh_post = self._normalise(vh.copy(), "VH_post")
            ratio   = self._normalise(vv - vh, "VV_VH_ratio")
        else:
            vv_pre, vh_pre, vv_post, vh_post = vv, vh, vv, vh
            ratio = vv - vh

        # Optional HAND per-chip (if present); else zeros
        if self.include_hand and s["hand_path"] is not None:
            hand = self._read_tif(Path(s["hand_path"]))  # (1,H,W)
            if self.normalize:
                hand = self._normalise(hand, "HAND")
        else:
            hand = np.zeros_like(vv)

        # Optional pop per-chip (if present); else zeros
        # (No normalization here unless you explicitly add pop stats later.)
        if self.include_pop and s["pop_path"] is not None:
            pop = self._read_tif(Path(s["pop_path"]))  # (1,H,W)
        else:
            pop = np.zeros_like(vv)

        # Label
        label_raw = self._read_tif(Path(s["label_path"]))  # (1,H,W)
        label = self._handle_permanent_water(label_raw)

        # 6-channel model input (WorldPop excluded)
        image = np.concatenate([vv_pre, vh_pre, vv_post, vh_post, ratio, hand], axis=0).astype(np.float32)

        # Crop (train only)
        if self.patch_size and self.split == "train":
            image, label = self._random_crop(image, label, size=self.patch_size)

        # Augment (train only)
        if self.augment:
            image, label = self._augment(image, label)

        # Cleanup
        image = np.nan_to_num(image, nan=0.0, posinf=5.0, neginf=-5.0)

        return {
            "image":    torch.from_numpy(image),            # (6,H,W)
            "label":    torch.from_numpy(label[0]).long(),  # (H,W)
            "hand_raw": torch.from_numpy(hand[0]),          # (H,W)
            "pop":      torch.from_numpy(pop[0]),           # (H,W)  (post-pred only)
            "chip_id":  s["chip_id"],
            "event":    s["event"],
        }

    def __len__(self) -> int:
        return len(self.samples)


# ─────────────────────────────────────────────
# Normalisation stats computation (optional utility)
# ─────────────────────────────────────────────

def compute_normalization_stats(data_root: str, split: str = "train") -> dict:
    """
    Optional: compute per-band mean/std for the 6-band model input.
    Saves to data_root/norm_stats.json.
    """
    ds = FloodDataset(data_root, split=split, patch_size=None, augment=False, normalize=False)

    band_names = ["VV_pre", "VH_pre", "VV_post", "VH_post", "VV_VH_ratio", "HAND"]
    sums = np.zeros(6, dtype=np.float64)
    sums2 = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.float64)

    print("Computing normalization statistics...")
    for i in range(len(ds)):
        sample = ds[i]
        img = sample["image"].numpy()  # (6,H,W)
        for b in range(6):
            x = img[b]
            mask = np.isfinite(x)
            v = x[mask]
            sums[b] += v.sum()
            sums2[b] += (v ** 2).sum()
            counts[b] += v.size

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(ds)} chips")

    mean = sums / np.maximum(counts, 1.0)
    var = sums2 / np.maximum(counts, 1.0) - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))

    stats = {name: (float(m), float(s)) for name, m, s in zip(band_names, mean, std)}
    out_path = Path(data_root) / "norm_stats.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"\nNorm stats saved → {out_path}")
    print(json.dumps(stats, indent=2))
    return stats


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    patch_size: int = 256,
    pin_memory: bool = True,
    oversample: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Train: random crops, augmentation, optional oversampling
    Val:   full chips
    Test:  Bolivia only (OOD)
    """
    train_ds = FloodDataset(data_root, split="train", patch_size=patch_size)
    val_ds   = FloodDataset(data_root, split="val",   patch_size=None, augment=False)
    test_ds  = FloodDataset(data_root, split="test",  patch_size=None, augment=False)

    if oversample:
        weights = []
        for s in train_ds.samples:
            try:
                with rasterio.open(s["label_path"]) as src:
                    label = src.read(1)
                valid = label[label != LABEL_IGNORE]
                flood_frac = float((valid == LABEL_FLOOD).mean()) if valid.size > 0 else 0.0
                # Weight proportional to flood content: chips richer in flood pixels
                # are sampled more often. Floor of 1.0 ensures all chips are seen.
                # Multiplier of 20 calibrated so a 5%-flood chip gets weight ~2.0.
                weights.append(1.0 + flood_frac * 20.0)
            except Exception:
                weights.append(1.0)

        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print("\nDataLoaders ready:")
    print(f"  Train : {len(train_ds):4d} chips  ({batch_size} per batch)")
    print(f"  Val   : {len(val_ds):4d} chips")
    print(f"  Test  : {len(test_ds):4d} chips  ← Bolivia OOD holdout")

    return train_loader, val_loader, test_loader
    
if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/sen1floods11"

    print("=== Sanity Check ===")
    train_loader, _, _ = get_dataloaders(data_root, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))

    print("\nSample batch:")
    print(f"  image shape : {batch['image'].shape}")   # (4, 6, 256, 256)
    print(f"  label shape : {batch['label'].shape}")   # (4, 256, 256)
    print(f"  hand  shape : {batch['hand_raw'].shape}")
    print(f"  events      : {batch['event']}")
    print(f"  label unique: {batch['label'].unique().tolist()}")
    print("\nAll good. Ready.")