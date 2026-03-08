"""
tools/alpha_vs_hand.py
======================
Gate attention weight α vs HAND elevation — validation of physics gate.

Analyses whether the HANDAttentionGate produces physically meaningful
attention maps: α should decrease with increasing HAND (height above
nearest drainage), as flood probability decays exponentially with elevation.

Theoretical curve (from gate design):
  α_gate(h) = exp(−h / 50.0)   [physics prior; h in metres]

Empirically extracted:
  α_empirical = mean gate attention per pixel, binned by HAND elevation.

Agreement between theoretical and empirical confirms the gate has learned
the physics prior from data rather than ignoring it.

Usage:
  python tools/alpha_vs_hand.py \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11 \\
      --output_dir results/alpha_vs_hand

Output:
  alpha_vs_hand.json         — binned α values vs HAND
  alpha_vs_hand.png          — empirical curve vs theoretical exp(−h/50)
  alpha_vs_hand_scatter.png  — pixel-level scatter (sample of N chips)
"""

import json
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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

from model   import build_model, GateAttentionHook   # noqa: E402
from dataset import get_dataloaders                   # noqa: E402


# ─── Gate extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_gate_vs_hand(
    ckpt_path:   str,
    data_root:   str,
    device:      torch.device,
    split:       str  = "test",
    n_chips:     int  = 15,
    batch_size:  int  = 1,
    num_workers: int  = 4,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Runs forward_with_gates() on each chip; collects all valid pixels'
    (HAND_metres, alpha_mean_across_4_gates) pairs.

    Args:
        n_chips: max chips to process (keep small for speed)

    Returns:
        hand_m:  (N,) float32 — per-pixel HAND in metres
        alpha:   (N,) float32 — mean attention across 4 gate levels
        variant: model variant name
    """
    ckpt    = torch.load(ckpt_path, map_location=device)
    config  = ckpt.get("config", {})
    variant = config.get("variant", "D")

    model = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    if not model.use_hand_gate:
        raise ValueError(
            f"Variant {variant} has no HAND gate (use_hand_gate=False). "
            "Use Variant C, D, D_plus, or E."
        )

    _, val_loader, test_loader = get_dataloaders(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers
    )
    loader = test_loader if split == "test" else val_loader

    HAND_MEAN_M = 9.346
    HAND_STD_M  = 28.330

    all_hand:  list[np.ndarray] = []
    all_alpha: list[np.ndarray] = []

    for i, batch in enumerate(loader):
        if i >= n_chips:
            break

        images = batch["image"].to(device)   # (B, 6, H, W)
        labels = batch["label"]              # (B, H, W)

        B, _, H, W = images.shape

        # Get attention maps via forward_with_gates
        logits, gate_maps = model.forward_with_gates(images)

        # gate_maps: list of 4 np arrays, each (B, H, W)
        # Stack and mean across 4 scales
        alpha_stack = np.stack(gate_maps, axis=0)   # (4, B, H, W)
        alpha_mean  = alpha_stack.mean(axis=0)       # (B, H, W)

        # HAND in metres
        hand_z = images[:, 5, :, :].cpu().numpy()   # (B, H, W)
        hand_m = hand_z * HAND_STD_M + HAND_MEAN_M

        for b in range(B):
            lab = labels[b].numpy()
            valid = lab != -1
            all_hand.append(hand_m[b][valid].flatten())
            all_alpha.append(alpha_mean[b][valid].flatten())

    if not all_hand:
        return np.array([]), np.array([]), variant

    hand_arr  = np.concatenate(all_hand).astype(np.float32)
    alpha_arr = np.concatenate(all_alpha).astype(np.float32)

    return hand_arr, alpha_arr, variant


# ─── Binning ───────────────────────────────────────────────────────────────

def bin_alpha_by_hand(
    hand:    np.ndarray,
    alpha:   np.ndarray,
    n_bins:  int = 20,
    hand_max: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bins pixels by HAND elevation and computes mean ± std of α per bin.

    Args:
        hand:     (N,) HAND in metres
        alpha:    (N,) attention weight
        n_bins:   number of HAND bins
        hand_max: clip HAND values above this (to focus on relevant range)

    Returns:
        bin_centres: (n_bins,) centre of each HAND bin
        mean_alpha:  (n_bins,) mean α per bin
        std_alpha:   (n_bins,) std  α per bin
    """
    hand  = np.clip(hand,  0.0, hand_max)
    edges = np.linspace(0.0, hand_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    means = np.full(n_bins, np.nan)
    stds  = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (hand >= edges[i]) & (hand < edges[i + 1])
        if mask.sum() > 10:
            means[i] = float(alpha[mask].mean())
            stds[i]  = float(alpha[mask].std())

    return centres, means, stds


# ─── Plots ─────────────────────────────────────────────────────────────────

def plot_alpha_vs_hand(
    centres:     np.ndarray,
    mean_alpha:  np.ndarray,
    std_alpha:   np.ndarray,
    variant:     str,
    out_path:    str,
) -> None:
    """
    Plots empirical α vs HAND alongside theoretical exp(−h/50).
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Theoretical curve
    h_theory = np.linspace(0, centres[-1] if len(centres) > 0 else 150, 300)
    ax.plot(h_theory, np.exp(-h_theory / 50.0),
            "k--", linewidth=2, label="Theory: exp(−h/50)")

    # Empirical binned mean
    valid = ~np.isnan(mean_alpha)
    ax.plot(centres[valid], mean_alpha[valid],
            "o-", color="#d62728", linewidth=2, markersize=5,
            label=f"Empirical (Variant {variant})")

    # Error band (±1 std)
    if std_alpha is not None and valid.sum() > 0:
        ax.fill_between(
            centres[valid],
            (mean_alpha - std_alpha)[valid].clip(0, 1),
            (mean_alpha + std_alpha)[valid].clip(0, 1),
            alpha=0.2, color="#d62728",
        )

    # Pearson r between empirical and theory
    interp_theory = np.exp(-centres[valid] / 50.0)
    if valid.sum() >= 3:
        r = float(np.corrcoef(mean_alpha[valid], interp_theory)[0, 1])
        ax.set_title(
            f"Variant {variant} — Gate attention α vs HAND elevation\n"
            f"Pearson r(empirical, theory) = {r:+.3f}",
            fontsize=11
        )
    else:
        ax.set_title(f"Variant {variant} — Gate attention α vs HAND elevation",
                     fontsize=11)

    ax.set_xlabel("HAND elevation (metres)", fontsize=11)
    ax.set_ylabel("Mean gate attention α", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate("Flood-prone\n(low HAND)", xy=(5, 0.9), fontsize=8,
                color="gray", ha="left")
    ax.annotate("Non-flood\n(high HAND)", xy=(120, 0.1), fontsize=8,
                color="gray", ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scatter_sample(
    hand:     np.ndarray,
    alpha:    np.ndarray,
    variant:  str,
    out_path: str,
    max_pts:  int = 50_000,
) -> None:
    """
    Pixel-level scatter of α vs HAND with theoretical overlay.
    Subsamples to max_pts for speed.
    """
    if len(hand) > max_pts:
        idx  = np.random.choice(len(hand), max_pts, replace=False)
        hand  = hand[idx]
        alpha = alpha[idx]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(hand.clip(0, 150), alpha, s=1, alpha=0.05, color="#1f77b4",
               rasterized=True)

    h_th = np.linspace(0, 150, 300)
    ax.plot(h_th, np.exp(-h_th / 50.0), "r-", linewidth=2,
            label="Theory exp(−h/50)")

    ax.set_xlabel("HAND elevation (m)", fontsize=11)
    ax.set_ylabel("Gate attention α", fontsize=11)
    ax.set_title(f"Variant {variant} — Pixel-level α vs HAND (n={len(hand):,})",
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────

def run_analysis(args: argparse.Namespace) -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints (variants C, D, E, D_plus support gates)
    ckpt_paths: list[str] = []
    if args.checkpoints:
        ckpt_paths = args.checkpoints
    elif args.checkpoints_dir:
        ckpt_base = Path(args.checkpoints_dir)
        for v in ["C", "D", "E", "D_plus"]:
            cp = ckpt_base / f"variant_{v}" / "best.pt"
            if cp.exists():
                ckpt_paths.append(str(cp))
    elif args.checkpoint:
        ckpt_paths = [args.checkpoint]
    else:
        raise ValueError("Provide --checkpoint, --checkpoints, or --checkpoints_dir")

    all_results: dict[str, dict] = {}

    for ckpt_path in ckpt_paths:
        print(f"\nProcessing: {ckpt_path}")
        try:
            hand_m, alpha, variant = extract_gate_vs_hand(
                ckpt_path=ckpt_path,
                data_root=args.data_root,
                device=device,
                split=args.split,
                n_chips=args.n_chips,
                num_workers=args.num_workers,
            )
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        if len(hand_m) == 0:
            print(f"  No data extracted for Variant {variant}. Skipping.")
            continue

        print(f"  Extracted {len(hand_m):,} pixel observations")

        centres, mean_alpha, std_alpha = bin_alpha_by_hand(
            hand_m, alpha,
            n_bins=args.n_bins,
            hand_max=args.hand_max,
        )

        # Pearson r vs theory
        valid = ~np.isnan(mean_alpha)
        if valid.sum() >= 3:
            theory = np.exp(-centres[valid] / 50.0)
            r_vs_theory = float(np.corrcoef(mean_alpha[valid], theory)[0, 1])
        else:
            r_vs_theory = float("nan")

        print(f"  Pearson r(empirical α, theory) = {r_vs_theory:+.3f}")

        all_results[variant] = {
            "n_pixels":         len(hand_m),
            "pearson_r_theory": round(r_vs_theory, 4),
            "bin_centres_m":    [round(c, 1) for c in centres.tolist()],
            "mean_alpha":       [round(float(v), 4) if not np.isnan(v) else None
                                 for v in mean_alpha.tolist()],
            "std_alpha":        [round(float(v), 4) if not np.isnan(v) else None
                                 for v in std_alpha.tolist()],
        }

        # Plots
        plot_alpha_vs_hand(
            centres, mean_alpha, std_alpha,
            variant=variant,
            out_path=str(out_dir / f"alpha_vs_hand_{variant}.png"),
        )
        if args.scatter:
            plot_scatter_sample(
                hand_m, alpha,
                variant=variant,
                out_path=str(out_dir / f"alpha_vs_hand_scatter_{variant}.png"),
            )

    # Save JSON
    (out_dir / "alpha_vs_hand.json").write_text(
        json.dumps(all_results, indent=2)
    )

    # Summary
    print(f"\n{'Variant':<12} {'n_pixels':>10} {'r vs theory':>12}")
    print("-" * 38)
    for v, res in all_results.items():
        print(f"  {v:<10} {res['n_pixels']:>10,} {res['pearson_r_theory']:>+12.3f}")

    print(f"\nResults saved → {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate attention α vs HAND elevation validation"
    )
    p.add_argument("--checkpoint",      type=str, default=None,
                   help="Single checkpoint path")
    p.add_argument("--checkpoints",     nargs="+", default=None,
                   help="Multiple checkpoint paths")
    p.add_argument("--checkpoints_dir", type=str, default=None,
                   help="Dir containing variant_C/, variant_D/, etc.")
    p.add_argument("--data_root",  type=str, default="data/sen1floods11")
    p.add_argument("--output_dir", type=str, default="results/alpha_vs_hand")
    p.add_argument("--split",      type=str, default="test",
                   choices=["test", "val"])
    p.add_argument("--n_chips",    type=int, default=15,
                   help="Number of chips to process (15 = Bolivia test set)")
    p.add_argument("--n_bins",     type=int, default=20,
                   help="Number of HAND elevation bins")
    p.add_argument("--hand_max",   type=float, default=150.0,
                   help="Clip HAND above this value in metres (focus on relevant range)")
    p.add_argument("--scatter",    action="store_true",
                   help="Also save pixel-level scatter plots (large files)")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
