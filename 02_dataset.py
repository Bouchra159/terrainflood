"""
Phase 1 — Dataset Loader
=========================
Handles two data sources:

  Source A — Sen1Floods11 local chips (downloaded via gsutil)
             Labels come from here: hand-annotated flood masks
             Images:  Bolivia_103757_S1Hand.tif  (Sentinel-1 VV, VH)
             Labels:  Bolivia_103757_LabelHand.tif (0=no flood, 1=flood, 2=permanent water)

  Source B — GEE-exported TFRecord chips
             Full 7-band stacks: [VV_pre, VH_pre, VV_post, VH_post, ratio, HAND, pop_log]
             No labels included (labels come from Sen1Floods11 alignment)

This loader:
  - Reads Sen1Floods11 GeoTIFFs or TFRecords
  - Applies per-band normalisation (zero-mean, unit-variance from training stats)
  - Returns (image_tensor, label_tensor, hand_tensor, meta) tuples
  - Handles class imbalance reporting
  - Splits by flood event (Bolivia always held out for OOD testing)

Usage:
  from 02_dataset import FloodDataset, get_dataloaders
  train_loader, val_loader, test_loader = get_dataloaders(
      data_root="data/sen1floods11",
      batch_size=8,
      num_workers=4
  )
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import rasterio
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Band statistics for normalisation
# Computed from Sen1Floods11 training set
# Update NORM_STATS after first run with compute_normalization_stats()
# ─────────────────────────────────────────────
NORM_STATS = {
    # band_name: (mean, std)   — in physical units (dB for SAR, metres for HAND)
    # 6-band model input: [VV_pre, VH_pre, VV_post, VH_post, VV_VH_ratio, HAND]
    # WorldPop is NOT included here — it is loaded separately into batch["pop"]
    "VV_pre":       (-12.5,  4.5),
    "VH_pre":       (-19.5,  4.5),
    "VV_post":      (-12.5,  4.5),
    "VH_post":      (-19.5,  4.5),
    "VV_VH_ratio":  ( -7.0,  3.0),
    "HAND":         (  5.0, 10.0),   # metres above nearest drainage
}

# Sen1Floods11 split — Bolivia is ALWAYS test (out-of-distribution)
SPLIT_MAP = {
    "Bolivia":    "test",
    "Cambodia":   "train",
    "Canada":     "train",
    "DemRepCongo":"train",
    "Ecuador":    "val",
    "Ghana":      "train",
    "India":      "train",
    "Mekong":     "train",
    "Nigeria":    "train",
    "Paraguay":   "val",
    "Somalia":    "train",
}

# Sen1Floods11 label values
LABEL_NO_FLOOD       = 0
LABEL_FLOOD          = 1
LABEL_PERMANENT_WATER= 2
LABEL_IGNORE         = -1   # masked / invalid pixels


class FloodDataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 + HAND + WorldPop inputs.

    Directory structure expected:
    data_root/
      flood_events/
        Bolivia_103757_S1Hand.tif       ← Sentinel-1 VV+VH (2 bands)
        Bolivia_103757_LabelHand.tif    ← Label (1 band)
        Bolivia_103757_S2Hand.tif       ← Sentinel-2 (optional, not used here)
        ...
      hand_chips/                        ← GEE-exported HAND chips (optional)
        Bolivia_103757_HAND.tif
      pop_chips/                         ← GEE-exported WorldPop chips (optional)
        Bolivia_103757_pop.tif

    If hand_chips/ and pop_chips/ do not exist, HAND and population
    are set to zeros and excluded from the HAND gate (graceful degradation).
    """

    def __init__(
        self,
        data_root:       str,
        split:           str = "train",          # "train" | "val" | "test"
        patch_size:      int = 256,              # crop size during training
        augment:         bool = True,
        normalize:       bool = True,
        include_hand:    bool = True,
        include_pop:     bool = True,
        permanent_water: str  = "include",       # "include" | "exclude" | "flood"
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

        print(f"[FloodDataset] split={split}  chips={len(self.samples)}"
              f"  flood_ratio={self.class_weights['flood_ratio']:.3f}")

    # ─── sample discovery ───────────────────

    def _discover_samples(self) -> list[dict]:
        """Scans data_root and builds a list of chip dicts for this split."""
        event_dir = self.data_root / "flood_events"
        if not event_dir.exists():
            raise FileNotFoundError(
                f"Expected Sen1Floods11 chips at {event_dir}\n"
                f"Download with:  gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events {self.data_root}/"
            )

        samples = []
        for label_path in sorted(event_dir.glob("*_LabelHand.tif")):
            chip_id    = label_path.stem.replace("_LabelHand", "")
            event_name = chip_id.split("_")[0]

            if SPLIT_MAP.get(event_name, "train") != self.split:
                continue

            s1_path = event_dir / f"{chip_id}_S1Hand.tif"
            if not s1_path.exists():
                continue

            # Optional HAND and pop chips from GEE
            hand_path = self.data_root / "hand_chips"  / f"{chip_id}_HAND.tif"
            pop_path  = self.data_root / "pop_chips"   / f"{chip_id}_pop.tif"

            samples.append({
                "chip_id":    chip_id,
                "event":      event_name,
                "s1_path":    s1_path,
                "label_path": label_path,
                "hand_path":  hand_path  if hand_path.exists()  else None,
                "pop_path":   pop_path   if pop_path.exists()   else None,
            })

        if not samples:
            raise RuntimeError(
                f"No chips found for split='{self.split}' in {event_dir}\n"
                f"Check that SPLIT_MAP matches your downloaded events."
            )
        return samples

    # ─── class weight estimation ─────────────

    def _compute_class_weights(self) -> dict:
        """
        Quick pass over label files to estimate flood pixel ratio.
        Used for WeightedRandomSampler and loss weighting.
        """
        total = flood = 0
        for s in self.samples[:min(50, len(self.samples))]:   # sample 50 chips
            try:
                with rasterio.open(s["label_path"]) as src:
                    label = src.read(1)
                valid = label[label != LABEL_IGNORE]
                flood += (valid == LABEL_FLOOD).sum()
                total += valid.size
            except Exception:
                pass
        ratio = float(flood) / max(total, 1)
        # BCE weight: inverse frequency
        pos_weight = (1.0 - ratio) / max(ratio, 1e-6)
        return {"flood_ratio": ratio, "pos_weight": min(pos_weight, 20.0)}

    # ─── loading helpers ─────────────────────

    def _read_tif(self, path, bands=None) -> np.ndarray:
        """Reads a GeoTIFF and returns a (C, H, W) float32 array."""
        with rasterio.open(path) as src:
            if bands is None:
                data = src.read().astype(np.float32)
            else:
                data = src.read(bands).astype(np.float32)
        return data

    def _normalise(self, band_data: np.ndarray, band_name: str) -> np.ndarray:
        """Zero-mean, unit-variance normalisation using precomputed stats."""
        if band_name not in NORM_STATS:
            return band_data
        mean, std = NORM_STATS[band_name]
        return (band_data - mean) / (std + 1e-6)

    def _handle_permanent_water(self, label: np.ndarray) -> np.ndarray:
        """
        Permanent water pixels (class 2) handling:
          'include' → treat as flood (1)    [most common choice]
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
        if h <= size or w <= size:
            # Pad if smaller than patch size
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            arrays = tuple(
                np.pad(a, [(0,0),(0,pad_h),(0,pad_w)], mode="reflect")
                for a in arrays
            )
            h, w = arrays[0].shape[-2], arrays[0].shape[-1]

        top  = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
        return tuple(a[..., top:top+size, left:left+size] for a in arrays)

    def _augment(self, image: np.ndarray, label: np.ndarray) -> tuple:
        """
        Training augmentations — all label-preserving:
          - Random horizontal flip
          - Random vertical flip
          - Random 90° rotation
        Note: SAR-specific — no colour jitter or brightness changes.
        """
        if np.random.random() > 0.5:
            image = np.flip(image, axis=-1).copy()
            label = np.flip(label, axis=-1).copy()
        if np.random.random() > 0.5:
            image = np.flip(image, axis=-2).copy()
            label = np.flip(label, axis=-2).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            label = np.rot90(label, k, axes=(-2, -1)).copy()
        return image, label

    # ─── __getitem__ ─────────────────────────

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # ── SAR: 2 bands (VV, VH) used as both pre and post
        # Sen1Floods11 provides single post-event S1 + S2Hand (optical-derived)
        # For bi-temporal: we use S1Hand as POST; pre-event pulled from GEE via
        # the export pipeline. For now, we use S1Hand for both (single-date baseline)
        # and extend to bi-temporal once GEE exports land.
        s1 = self._read_tif(s["s1_path"])  # shape: (2, H, W) — VV, VH

        # Build input stack: [VV_pre, VH_pre, VV_post, VH_post]
        # In single-date mode: pre = post (model still learns water signature)
        vv = s1[0:1]   # (1, H, W)
        vh = s1[1:2]   # (1, H, W)

        if self.normalize:
            vv_pre  = self._normalise(vv.copy(), "VV_pre")
            vh_pre  = self._normalise(vh.copy(), "VH_pre")
            vv_post = self._normalise(vv.copy(), "VV_post")
            vh_post = self._normalise(vh.copy(), "VH_post")
            ratio   = self._normalise(vv - vh,   "VV_VH_ratio")
        else:
            vv_pre = vh_pre = vv_post = vh_post = vv - vh
            vv_pre = vv; vh_pre = vh; vv_post = vv; vh_post = vh
            ratio  = vv - vh

        # ── HAND band
        if self.include_hand and s["hand_path"] is not None:
            hand = self._read_tif(s["hand_path"])  # (1, H, W)
            if self.normalize:
                hand = self._normalise(hand, "HAND")
        else:
            hand = np.zeros_like(vv)

        # ── Population band
        if self.include_pop and s["pop_path"] is not None:
            pop = self._read_tif(s["pop_path"])   # (1, H, W)
            if self.normalize:
                pop = self._normalise(pop, "pop_log")
        else:
            pop = np.zeros_like(vv)

        # ── Label
        label_raw = self._read_tif(s["label_path"])  # (1, H, W)
        label     = self._handle_permanent_water(label_raw)

        # Stack image: 6 channels total (WorldPop excluded from model input)
        # [VV_pre, VH_pre, VV_post, VH_post, ratio, HAND]
        image = np.concatenate([vv_pre, vh_pre, vv_post, vh_post, ratio, hand],
                               axis=0).astype(np.float32)

        # ── Random crop
        if self.patch_size and self.split == "train":
            image, label = self._random_crop(image, label, size=self.patch_size)

        # ── Augmentation
        if self.augment:
            image, label = self._augment(image, label)

        # NaN/inf cleanup
        image = np.nan_to_num(image, nan=0.0, posinf=5.0, neginf=-5.0)

        return {
            "image":    torch.from_numpy(image),           # (6, H, W) float32
            "label":    torch.from_numpy(label[0]).long(),  # (H, W)    int64
            "hand_raw": torch.from_numpy(hand[0]),          # (H, W)    float32 — raw HAND for gate
            "pop":      torch.from_numpy(pop[0]),           # (H, W)    float32 — WorldPop, for 06_exposure.py only
            "chip_id":  s["chip_id"],
            "event":    s["event"],
        }

    def __len__(self) -> int:
        return len(self.samples)


# ─────────────────────────────────────────────
# Normalisation stats computation
# ─────────────────────────────────────────────

def compute_normalization_stats(data_root: str, split: str = "train") -> dict:
    """
    Run this ONCE after downloading data to compute per-band statistics.
    Saves results to data_root/norm_stats.json.
    """
    ds = FloodDataset(data_root, split=split, patch_size=None,
                      augment=False, normalize=False)

    band_names = ["VV_pre", "VH_pre", "VV_post", "VH_post",
                  "VV_VH_ratio", "HAND"]
    sums  = np.zeros(6)
    sums2 = np.zeros(6)
    count = 0

    print("Computing normalization statistics...")
    for i, sample in enumerate(ds):
        img = sample["image"].numpy()  # (6, H, W)
        mask = np.isfinite(img)
        for b in range(6):
            valid = img[b][mask[b]]
            sums[b]  += valid.sum()
            sums2[b] += (valid ** 2).sum()
            count    += valid.size
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(ds)} chips")

    n    = count / 6   # per-band count
    mean = sums  / n
    std  = np.sqrt(sums2/n - mean**2)

    stats = {name: (float(m), float(s))
             for name, m, s in zip(band_names, mean, std)}

    out_path = Path(data_root) / "norm_stats.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"\nNorm stats saved → {out_path}")
    print(json.dumps(stats, indent=2))
    return stats


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    data_root:      str,
    batch_size:     int  = 8,
    num_workers:    int  = 4,
    patch_size:     int  = 256,
    pin_memory:     bool = True,
    oversample:     bool = True,   # WeightedRandomSampler to handle class imbalance
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Train: random crops, augmentation, optional oversampling
    Val:   no crops/augmentation, full chips
    Test:  Bolivia only, no augmentation (OOD evaluation)
    """
    train_ds = FloodDataset(data_root, split="train", patch_size=patch_size)
    val_ds   = FloodDataset(data_root, split="val",   patch_size=None, augment=False)
    test_ds  = FloodDataset(data_root, split="test",  patch_size=None, augment=False)

    # Weighted sampler: oversample flood-containing chips
    if oversample:
        weights = []
        for s in train_ds.samples:
            try:
                with rasterio.open(s["label_path"]) as src:
                    label = src.read(1)
                has_flood = int((label == LABEL_FLOOD).any())
                weights.append(2.0 if has_flood else 1.0)
            except Exception:
                weights.append(1.0)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"\nDataLoaders ready:")
    print(f"  Train : {len(train_ds):4d} chips  ({batch_size} per batch)")
    print(f"  Val   : {len(val_ds):4d} chips")
    print(f"  Test  : {len(test_ds):4d} chips  ← Bolivia OOD holdout")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/sen1floods11"

    print("=== Sanity Check ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root, batch_size=4, num_workers=0
    )

    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  image shape : {batch['image'].shape}")   # (4, 7, 256, 256)
    print(f"  label shape : {batch['label'].shape}")   # (4, 256, 256)
    print(f"  hand  shape : {batch['hand_raw'].shape}")
    print(f"  events      : {batch['event']}")
    print(f"  label unique: {batch['label'].unique().tolist()}")
    print(f"\nAll good. Ready for Phase 2.")
