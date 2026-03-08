"""
tools/iou_vs_hand.py
====================
Per-chip IoU vs mean HAND elevation analysis.

Answers: "Does model performance degrade in high-HAND (hillside) terrain?"
This validates the physics gate contribution: with a well-trained HAND gate,
high-HAND regions should be suppressed (low false positives) and performance
should not degrade as severely as in Variants A/B.

Method:
  1. Run inference on the test set (Bolivia OOD) for each variant.
  2. Compute per-chip IoU and mean HAND elevation.
  3. Compute Pearson r between IoU and HAND.
  4. Plot scatter with regression line + HAND-binned bar chart.

Usage:
  python tools/iou_vs_hand.py \\
      --checkpoints_dir checkpoints \\
      --data_root data/sen1floods11 \\
      --output_dir results/iou_vs_hand

Output:
  iou_vs_hand.json       — per-chip data for all variants
  iou_vs_hand_scatter.png — scatter plot
  iou_vs_hand_binned.png  — binned bar chart (HAND quantiles)
"""

import json
import argparse
import importlib.util
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ─── Module imports ────────────────────────────────────────────────────────

_root = Path(__file__).parent.parent


def _import_module(alias: str, path: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


if "model"   not in sys.modules:
    _import_module("model",   str(_root / "03_model.py"))
if "dataset" not in sys.modules:
    _import_module("dataset", str(_root / "02_dataset.py"))

from model   import build_model      # noqa: E402
from dataset import get_dataloaders  # noqa: E402


# ─── Per-chip inference ────────────────────────────────────────────────────

@torch.no_grad()
def chip_inference(
    ckpt_path:   str,
    data_root:   str,
    device:      torch.device,
    split:       str = "test",
    batch_size:  int = 1,
    num_workers: int = 4,
    threshold:   float = 0.5,
) -> tuple[list[dict], str]:
    """
    Runs single-pass inference. Returns per-chip results and variant name.

    Each result dict has:
      chip_id, event, iou, mean_hand_m (metres, denormalised)
    """
    ckpt    = torch.load(ckpt_path, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    _, val_loader, test_loader = get_dataloaders(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers
    )
    loader = test_loader if split == "test" else val_loader

    # HAND denormalisation constants (from norm_stats.json training split)
    HAND_MEAN_M = 9.346
    HAND_STD_M  = 28.330

    results: list[dict] = []

    for batch in loader:
        images   = batch["image"].to(device)    # (B, 6, H, W)
        labels   = batch["label"]               # (B, H, W)
        hand_z   = batch["image"][:, 5, :, :]  # z-score HAND

        logits = model(images).squeeze(1).cpu().numpy()   # (B, H, W)
        probs  = (1.0 / (1.0 + np.exp(-logits)))

        for i in range(images.shape[0]):
            lab  = labels[i].numpy()
            valid = lab != -1
            if valid.sum() == 0:
                continue

            p = probs[i][valid].flatten()
            l = lab[valid].flatten()

            preds = (p > threshold).astype(np.int32)
            l_int = l.astype(np.int32)

            tp = int(((preds == 1) & (l_int == 1)).sum())
            fp = int(((preds == 1) & (l_int == 0)).sum())
            fn = int(((preds == 0) & (l_int == 1)).sum())
            iou = tp / max(tp + fp + fn, 1)

            # HAND: denormalise z-score → metres, then take mean over valid pixels
            hz = hand_z[i].numpy()
            hand_m = hz[valid].flatten() * HAND_STD_M + HAND_MEAN_M
            mean_hand = float(hand_m.mean())

            chip_id = batch["chip_id"][i]
            event   = batch["event"][i]

            results.append({
                "chip_id":     chip_id,
                "event":       event,
                "iou":         round(iou, 4),
                "mean_hand_m": round(mean_hand, 2),
            })

    return results, variant


# ─── Pearson correlation ────────────────────────────────────────────────────

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between x and y (both 1-D arrays)."""
    if len(x) < 3:
        return float("nan")
    corr = np.corrcoef(x, y)[0, 1]
    return round(float(corr), 4)


# ─── Plots ─────────────────────────────────────────────────────────────────

def plot_scatter(
    data:     dict[str, list[dict]],
    out_path: str,
) -> None:
    """Scatter: mean HAND (m) vs IoU, one subplot per variant."""
    variants = list(data.keys())
    n        = len(variants)
    ncols    = min(n, 3)
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    COLORS = {
        "A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728",
        "E": "#9467bd", "D_plus": "#8c564b", "baseline_unet": "#7f7f7f",
    }

    for idx, variant in enumerate(variants):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        chips = data[variant]

        hands = np.array([c["mean_hand_m"] for c in chips])
        ious  = np.array([c["iou"]         for c in chips])
        r     = pearson_r(hands, ious)
        color = COLORS.get(variant, "steelblue")

        ax.scatter(hands, ious, color=color, alpha=0.8, s=60, edgecolors="white")

        # Regression line
        if len(hands) >= 3:
            m, b = np.polyfit(hands, ious, 1)
            x_line = np.linspace(hands.min(), hands.max(), 100)
            ax.plot(x_line, m * x_line + b, "--", color=color, linewidth=1.5)

        ax.set_xlabel("Mean HAND (m)", fontsize=10)
        ax.set_ylabel("IoU", fontsize=10)
        ax.set_title(f"Variant {variant}\nr = {r:+.3f}  n={len(chips)}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(variants), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("IoU vs Mean HAND Elevation per Chip", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_binned(
    data:      dict[str, list[dict]],
    out_path:  str,
    n_bins:    int = 4,
) -> None:
    """
    Binned bar chart: HAND quantile bins on x-axis, mean IoU per bin per variant.
    Visualises whether IoU degrades in high-HAND (hillside) terrain.
    """
    # Compute HAND quantile bin edges from all chips (pooled)
    all_hand = np.concatenate([
        [c["mean_hand_m"] for c in chips]
        for chips in data.values()
    ])
    bin_edges = np.percentile(all_hand, np.linspace(0, 100, n_bins + 1))
    bin_labels = [
        f"{bin_edges[i]:.0f}–{bin_edges[i+1]:.0f} m"
        for i in range(n_bins)
    ]

    COLORS = {
        "A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728",
        "E": "#9467bd", "D_plus": "#8c564b", "baseline_unet": "#7f7f7f",
    }

    variants = list(data.keys())
    n_vars   = len(variants)
    x        = np.arange(n_bins)
    width    = 0.8 / n_vars

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, variant in enumerate(variants):
        chips = data[variant]
        bin_ious = []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            subset = [
                c["iou"] for c in chips
                if lo <= c["mean_hand_m"] < hi
            ]
            bin_ious.append(float(np.mean(subset)) if subset else float("nan"))

        offset = (i - n_vars / 2 + 0.5) * width
        ax.bar(x + offset, bin_ious, width=width * 0.9,
               color=COLORS.get(variant, "steelblue"),
               alpha=0.85, label=f"Variant {variant}")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Mean HAND elevation bin", fontsize=11)
    ax.set_ylabel("Mean IoU", fontsize=11)
    ax.set_title("IoU by HAND Elevation Quartile", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────

def run_analysis(args: argparse.Namespace) -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints
    ckpt_paths: list[str] = []
    if args.checkpoints:
        ckpt_paths = args.checkpoints
    elif args.checkpoints_dir:
        ckpt_base = Path(args.checkpoints_dir)
        for v in ["A", "B", "C", "D", "E", "D_plus", "baseline_unet"]:
            cp = ckpt_base / f"variant_{v}" / "best.pt"
            if cp.exists():
                ckpt_paths.append(str(cp))
    else:
        raise ValueError("Provide --checkpoints or --checkpoints_dir")

    all_data:     dict[str, list[dict]] = {}
    summary:      dict[str, dict]       = {}

    for ckpt_path in ckpt_paths:
        print(f"\nProcessing: {ckpt_path}")
        chips, variant = chip_inference(
            ckpt_path=ckpt_path,
            data_root=args.data_root,
            device=device,
            split=args.split,
            num_workers=args.num_workers,
        )
        all_data[variant] = chips

        hands = np.array([c["mean_hand_m"] for c in chips])
        ious  = np.array([c["iou"]         for c in chips])
        r     = pearson_r(hands, ious)
        mean_iou = float(ious.mean())
        summary[variant] = {
            "n_chips":       len(chips),
            "pearson_r":     r,
            "mean_iou":      round(mean_iou, 4),
            "mean_hand_m":   round(float(hands.mean()), 1),
        }
        print(f"  Variant {variant}: n={len(chips)}  "
              f"mean IoU={mean_iou:.4f}  Pearson r(IoU,HAND)={r:+.3f}")

    # Save data
    (out_dir / "iou_vs_hand.json").write_text(
        json.dumps({"summary": summary, "per_chip": all_data}, indent=2)
    )

    # Plots
    plot_scatter(all_data, str(out_dir / "iou_vs_hand_scatter.png"))
    if len(all_data) > 0:
        plot_binned(all_data, str(out_dir / "iou_vs_hand_binned.png"),
                    n_bins=args.n_bins)

    # Summary table
    print(f"\n{'Variant':<12} {'n':>4} {'IoU':>8} {'HAND mean':>10} {'Pearson r':>10}")
    print("-" * 50)
    for v, s in summary.items():
        print(f"  {v:<10} {s['n_chips']:>4} {s['mean_iou']:>8.4f} "
              f"{s['mean_hand_m']:>10.1f} {s['pearson_r']:>+10.3f}")

    print(f"\nResults saved → {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-chip IoU vs mean HAND elevation analysis"
    )
    p.add_argument("--checkpoints",     nargs="+", default=None,
                   help="Explicit checkpoint paths")
    p.add_argument("--checkpoints_dir", type=str,  default=None,
                   help="Dir containing variant_A/, variant_B/, etc.")
    p.add_argument("--data_root",  type=str, default="data/sen1floods11")
    p.add_argument("--output_dir", type=str, default="results/iou_vs_hand")
    p.add_argument("--split",      type=str, default="test",
                   choices=["test", "val"])
    p.add_argument("--n_bins",     type=int, default=4,
                   help="Number of HAND quantile bins for bar chart")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
