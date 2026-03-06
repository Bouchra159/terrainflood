#!/usr/bin/env python3
"""
HAND Attention Gate Visualisation
File: tools/visualize_gate.py

Loads a Variant C or D checkpoint, runs inference on the test (Bolivia OOD)
or validation set, extracts the 4-level HAND attention gate maps via
FloodSegmentationModel.forward_with_gates(), and saves publication-quality
figures using plots.plot_hand_gate_maps().

Key interpretation:
  α → 1.0 (bright green)  = low HAND / near-river  → flood signal retained
  α → 0.0 (dark red)      = high HAND / hillside    → features suppressed

Usage:
  # Default: Variant D checkpoint, Bolivia test set, 15 chips
  python tools/visualize_gate.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/gate_maps \\
      --split      test \\
      --n_chips    15

  # Variant C with validation set
  python tools/visualize_gate.py \\
      --checkpoint checkpoints/variant_C/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/gate_maps_C \\
      --split      val \\
      --n_chips    10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────
# Numeric-prefix imports (mirrors eval.py)
# ─────────────────────────────────────────────────────────────

_root = Path(__file__).parent.parent   # project root


def _import_module(alias: str, file_path: str):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "model_gate"   not in sys.modules:
    _import_module("model_gate",   str(_root / "03_model.py"))
if "dataset_gate" not in sys.modules:
    _import_module("dataset_gate", str(_root / "02_dataset.py"))

from model_gate   import build_model      # noqa: E402
from dataset_gate import get_dataloaders  # noqa: E402

# plots.py is in project root
sys.path.insert(0, str(_root))
from plots import plot_hand_gate_maps     # noqa: E402


# ─────────────────────────────────────────────────────────────
# Constants (must match _prepare_hand in 03_model.py)
# ─────────────────────────────────────────────────────────────
_HAND_MEAN: float = 9.346   # metres — from norm_stats.json
_HAND_STD:  float = 28.330  # metres — from norm_stats.json


def _denorm_hand(hand_z: np.ndarray) -> np.ndarray:
    """Denormalise z-scored HAND back to metres for display."""
    return hand_z * _HAND_STD + _HAND_MEAN


# ─────────────────────────────────────────────────────────────
# Core visualisation loop
# ─────────────────────────────────────────────────────────────

def visualise_gates(
    ckpt_path:    str,
    data_root:    str,
    output_dir:   str,
    device:       torch.device,
    split:        str  = "test",
    n_chips:      int  = 15,
    batch_size:   int  = 1,
    num_workers:  int  = 2,
) -> None:
    """
    Run gate visualisation on `n_chips` chips from the requested split.

    Args:
        ckpt_path:   path to best.pt checkpoint (Variant C or D only)
        data_root:   Sen1Floods11 root directory
        output_dir:  directory to save gate map figures
        device:      torch device
        split:       "test" (Bolivia OOD) or "val"
        n_chips:     number of chips to visualise
        batch_size:  batch size for inference (1 recommended for clarity)
        num_workers: dataloader workers
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────
    ckpt    = torch.load(ckpt_path, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    if variant not in ("C", "D"):
        raise ValueError(
            f"Gate visualisation only makes sense for Variants C and D "
            f"(use_hand_gate=True). Got Variant {variant}."
        )

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # For Variant D: keep dropout active to match MC inference behaviour
    # (gate maps are deterministic — gate uses no dropout — but we stay
    #  consistent with the inference setup)
    if variant == "D":
        model.enable_dropout()

    print(f"Loaded Variant {variant}  epoch={ckpt['epoch']}  "
          f"best_iou={ckpt.get('best_iou', 0.0):.4f}")
    print(f"Split: {split}  |  Chips to visualise: {n_chips}")
    print(f"Output: {out_dir}\n")

    # ── Dataloader ───────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root   = data_root,
        batch_size  = batch_size,
        num_workers = num_workers,
    )
    loader = test_loader if split == "test" else val_loader

    # ── Inference loop ───────────────────────────────────────
    n_saved  = 0
    metadata = []   # collect per-chip summary for JSON

    with torch.no_grad():
        for batch in loader:
            if n_saved >= n_chips:
                break

            images = batch["image"].to(device)     # (B, 6, H, W)
            labels = batch["label"].numpy()        # (B, H, W)
            events = batch["event"]                # list of strings
            chips  = batch["chip_id"]              # list of strings

            # forward_with_gates() — returns (logits, gate_maps)
            logits, gate_maps = model.forward_with_gates(images)

            # gate_maps is list of 4 arrays (B, H, W)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)

            for b in range(images.shape[0]):
                if n_saved >= n_chips:
                    break

                chip_id  = chips[b]
                event    = events[b]
                label_b  = labels[b]    # (H, W)
                prob_b   = probs[b]     # (H, W)

                # Extract SAR VV post-event (channel 2)
                sar_vv_b = images[b, 2, :, :].cpu().numpy()    # (H, W) dB

                # Denormalise HAND to metres (channel 5)
                hand_z_b = images[b, 5, :, :].cpu().numpy()    # (H, W) z-score
                hand_m_b = _denorm_hand(hand_z_b)              # (H, W) metres

                # Gate maps: list of 4 arrays (H, W) for this batch element
                gate_b = [gm[b] for gm in gate_maps]

                # Save figure
                fname = f"{event}_{chip_id}_gate.png"
                plot_hand_gate_maps(
                    sar_vv    = sar_vv_b,
                    hand_m    = hand_m_b,
                    gate_maps = gate_b,
                    label     = label_b,
                    mean_prob = prob_b,
                    chip_id   = chip_id,
                    out_path  = str(out_dir / fname),
                )

                # Collect summary stats
                valid_mask = label_b != -1
                gate_mean_finest   = float(gate_b[0][valid_mask].mean())
                gate_mean_coarsest = float(gate_b[3][valid_mask].mean())
                hand_mean          = float(hand_m_b[valid_mask].mean())

                metadata.append({
                    "chip_id":          chip_id,
                    "event":            event,
                    "hand_mean_m":      round(hand_mean, 2),
                    "gate_alpha0_mean": round(gate_mean_finest,   4),
                    "gate_alpha3_mean": round(gate_mean_coarsest, 4),
                    "figure":           fname,
                })

                n_saved += 1
                print(f"  [{n_saved:2d}/{n_chips}] {chip_id:30s}  "
                      f"HAND={hand_mean:.1f}m  "
                      f"α₀={gate_mean_finest:.3f}  "
                      f"α₃={gate_mean_coarsest:.3f}")

    # Save JSON summary
    summary_path = out_dir / "gate_summary.json"
    summary_path.write_text(json.dumps({
        "variant":    variant,
        "checkpoint": ckpt_path,
        "split":      split,
        "n_chips":    n_saved,
        "chips":      metadata,
    }, indent=2))
    print(f"\nGate summary → {summary_path}")
    print(f"Saved {n_saved} gate map figures to {out_dir}")

    # Print a quick physics check: high-HAND chips should have low mean α
    if metadata:
        sorted_by_hand = sorted(metadata, key=lambda r: r["hand_mean_m"])
        print("\nPhysics check — gate suppression by HAND:")
        print(f"  {'Chip':<30} {'HAND(m)':>8} {'α₀':>7} {'α₃':>7}")
        print("  " + "-" * 58)
        for r in sorted_by_hand[:5]:
            print(f"  {r['chip_id']:<30} {r['hand_mean_m']:>8.1f} "
                  f"{r['gate_alpha0_mean']:>7.3f} {r['gate_alpha3_mean']:>7.3f}")
        print("  ...")
        for r in sorted_by_hand[-3:]:
            print(f"  {r['chip_id']:<30} {r['hand_mean_m']:>8.1f} "
                  f"{r['gate_alpha0_mean']:>7.3f} {r['gate_alpha3_mean']:>7.3f}")
        print("  Expected: higher HAND → lower α (gate suppresses high-elevation features)")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise HAND attention gate maps (Variants C/D only)"
    )
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to best.pt (Variant C or D)")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--output_dir",   type=str, default="results/gate_maps",
                   help="Directory to save gate map figures")
    p.add_argument("--split",        type=str, default="test",
                   choices=["test", "val"],
                   help="test = Bolivia OOD (15 chips), val = Paraguay")
    p.add_argument("--n_chips",      type=int, default=15,
                   help="Number of chips to visualise")
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--num_workers",  type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    visualise_gates(
        ckpt_path   = args.checkpoint,
        data_root   = args.data_root,
        output_dir  = args.output_dir,
        device      = device,
        split       = args.split,
        n_chips     = args.n_chips,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
