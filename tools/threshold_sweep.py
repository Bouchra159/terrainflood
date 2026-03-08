"""
tools/threshold_sweep.py
========================
Sweeps the decision threshold τ from 0.20 to 0.80 and reports IoU, F1,
Precision, Recall per variant. Identifies the optimal τ that maximises IoU
on the test set (Bolivia OOD).

This answers reviewer question: "Is the 0.5 threshold optimal?"

Usage:
  python tools/threshold_sweep.py \\
      --results_dir results \\
      --variants D C B A \\
      --output_dir results/threshold_sweep

  Or pass raw mean_prob .npy arrays saved with --save_arrays:
  python tools/threshold_sweep.py \\
      --arrays_dir results/uncertainty/arrays \\
      --checkpoint checkpoints/variant_D/best.pt \\
      --data_root  data/sen1floods11

Output:
  threshold_sweep.json  — full sweep data per variant
  threshold_sweep.png   — IoU vs threshold curves (one line per variant)
  optimal_thresholds.json — per-variant τ* and corresponding metrics
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


# ─── Core sweep function ───────────────────────────────────────────────────

def sweep_thresholds(
    probs:       np.ndarray,
    labels:      np.ndarray,
    thresholds:  np.ndarray,
    ignore_value: int = -1,
) -> dict[str, list]:
    """
    Sweeps decision thresholds and returns metrics at each τ.

    Args:
        probs:       (N,) predicted flood probabilities
        labels:      (N,) int ground truth (0/1/-1)
        thresholds:  1-D array of τ values to evaluate
        ignore_value: label value to skip

    Returns:
        dict with keys: thresholds, iou, f1, precision, recall, accuracy
    """
    valid  = labels != ignore_value
    probs  = probs[valid].astype(np.float32)
    labels = labels[valid].astype(np.int32)

    metrics: dict[str, list] = {
        "thresholds": thresholds.tolist(),
        "iou":        [],
        "f1":         [],
        "precision":  [],
        "recall":     [],
        "accuracy":   [],
    }

    for tau in thresholds:
        preds = (probs > tau).astype(np.int32)

        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        iou       = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)

        metrics["iou"].append(round(iou, 4))
        metrics["f1"].append(round(f1, 4))
        metrics["precision"].append(round(precision, 4))
        metrics["recall"].append(round(recall, 4))
        metrics["accuracy"].append(round(accuracy, 4))

    return metrics


# ─── Load predictions from checkpoint ─────────────────────────────────────

def load_predictions_from_checkpoint(
    ckpt_path:   str,
    data_root:   str,
    device:      torch.device,
    split:       str = "test",
    batch_size:  int = 4,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Runs a single deterministic forward pass and returns (probs, labels, variant).
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

    all_probs:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images).squeeze(1).cpu().numpy()
            probs  = 1.0 / (1.0 + np.exp(-logits))

            for i in range(images.shape[0]):
                valid = labels[i].numpy() != -1
                all_probs.append(probs[i][valid].flatten())
                all_labels.append(labels[i].numpy()[valid].flatten())

    return (
        np.concatenate(all_probs).astype(np.float32),
        np.concatenate(all_labels).astype(np.int32),
        variant,
    )


# ─── Plot ──────────────────────────────────────────────────────────────────

def plot_sweep(
    sweep_data:  dict[str, dict],
    out_path:    str,
    metric:      str = "iou",
) -> None:
    """
    Plots metric vs threshold for each variant.
    Marks the optimal τ* with a vertical dashed line per variant.

    Args:
        sweep_data: {variant: metrics_dict}
        out_path:   output figure path
        metric:     which metric to plot on y-axis (default "iou")
    """
    COLORS = {
        "A":             "#1f77b4",
        "B":             "#ff7f0e",
        "C":             "#2ca02c",
        "D":             "#d62728",
        "E":             "#9467bd",
        "D_plus":        "#8c564b",
        "baseline_unet": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    for variant, data in sweep_data.items():
        taus   = np.array(data["thresholds"])
        values = np.array(data[metric])
        color  = COLORS.get(variant, "black")
        ax.plot(taus, values, label=f"Variant {variant}", color=color, linewidth=2)

        # Mark optimal τ
        best_idx = int(np.argmax(values))
        ax.axvline(taus[best_idx], color=color, linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.scatter([taus[best_idx]], [values[best_idx]],
                   color=color, s=60, zorder=5)

    ax.axvline(0.5, color="black", linestyle=":", linewidth=1, label="τ=0.5 default")
    ax.set_xlabel("Decision threshold τ", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Threshold sweep — {metric.upper()} vs τ", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────

def run_sweep(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.linspace(0.20, 0.80, 61)   # step = 0.01

    sweep_data:    dict[str, dict] = {}
    optimal_taus:  dict[str, dict] = {}

    # Load predictions — either from checkpoint list or arrays_dir
    ckpt_paths: list[tuple[str, str]] = []   # (ckpt_path, variant_override)

    if args.checkpoints:
        for ckpt in args.checkpoints:
            ckpt_paths.append((ckpt, ""))
    elif args.checkpoints_dir:
        ckpt_base = Path(args.checkpoints_dir)
        for v in ["A", "B", "C", "D", "E", "D_plus", "baseline_unet"]:
            cp = ckpt_base / f"variant_{v}" / "best.pt"
            if cp.exists():
                ckpt_paths.append((str(cp), v))
    else:
        raise ValueError("Provide --checkpoints or --checkpoints_dir")

    for ckpt_path, variant_hint in ckpt_paths:
        print(f"\nLoading: {ckpt_path}")
        probs, labels, variant = load_predictions_from_checkpoint(
            ckpt_path=ckpt_path,
            data_root=args.data_root,
            device=device,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if variant_hint:
            variant = variant_hint

        metrics = sweep_thresholds(probs, labels, thresholds)
        sweep_data[variant] = metrics

        best_idx = int(np.argmax(metrics["iou"]))
        opt_tau  = thresholds[best_idx]
        optimal_taus[variant] = {
            "tau_star":      round(float(opt_tau), 3),
            "iou_at_tau":    metrics["iou"][best_idx],
            "f1_at_tau":     metrics["f1"][best_idx],
            "iou_at_0.5":    metrics["iou"][int(round((0.5 - 0.20) / (0.60 / 60)))],
        }
        print(f"  Variant {variant}: τ* = {opt_tau:.2f}  "
              f"IoU@τ* = {metrics['iou'][best_idx]:.4f}  "
              f"IoU@0.5 = {optimal_taus[variant]['iou_at_0.5']:.4f}")

    # Save results
    (out_dir / "threshold_sweep.json").write_text(
        json.dumps(sweep_data, indent=2)
    )
    (out_dir / "optimal_thresholds.json").write_text(
        json.dumps(optimal_taus, indent=2)
    )

    # Plot
    plot_sweep(sweep_data, str(out_dir / "threshold_sweep_iou.png"), metric="iou")
    plot_sweep(sweep_data, str(out_dir / "threshold_sweep_f1.png"),  metric="f1")

    # Summary table
    print(f"\n{'Variant':<12} {'τ*':>6} {'IoU@τ*':>10} {'IoU@0.5':>10}")
    print("-" * 42)
    for v, info in optimal_taus.items():
        print(f"  {v:<10} {info['tau_star']:>6.3f} "
              f"{info['iou_at_tau']:>10.4f} {info['iou_at_0.5']:>10.4f}")

    print(f"\nResults saved → {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold sweep for flood segmentation")
    p.add_argument("--checkpoints",     nargs="+", default=None,
                   help="Explicit checkpoint paths (one or more best.pt files)")
    p.add_argument("--checkpoints_dir", type=str, default=None,
                   help="Directory containing variant_A/, variant_B/, etc.")
    p.add_argument("--data_root",  type=str, default="data/sen1floods11")
    p.add_argument("--output_dir", type=str, default="results/threshold_sweep")
    p.add_argument("--split",      type=str, default="test",
                   choices=["test", "val"])
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
