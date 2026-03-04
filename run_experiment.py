"""
End-to-End Experiment Runner
==============================
File: run_experiment.py

Trains all 4 ablation variants and runs the full downstream pipeline:
  1. Train variants A, B, C, D  (via train.py)
  2. Evaluate each variant       (via eval.py)
  3. Run MC uncertainty + calibration for Variant D  (via uncertainty.py)
  4. Run population exposure analysis for Variant D  (via 06_exposure.py)
  5. Build ablation comparison figures

Each step is skipped automatically if its output already exists,
so you can re-run safely after partial failures.

Usage:
  # Full run (all 4 variants × 50 epochs each — use on DKUCC)
  python run_experiment.py --data_root data/sen1floods11

  # Quick smoke test (2 batches per epoch, skips slow steps)
  python run_experiment.py --data_root data/sen1floods11 --fast

  # Skip training (use existing checkpoints), only evaluate
  python run_experiment.py --data_root data/sen1floods11 --eval_only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], desc: str) -> int:
    """
    Runs a subprocess command, streaming its output.
    Returns the exit code.
    """
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}")

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"\n  ✗ FAILED (exit {proc.returncode}): {desc}")
    else:
        print(f"\n  ✓ Done: {desc}")
    return proc.returncode


def python() -> str:
    """Returns the current Python interpreter path."""
    return sys.executable


def checkpoint_exists(ckpt_dir: Path) -> bool:
    """Returns True if best.pt exists in ckpt_dir."""
    return (ckpt_dir / "best.pt").exists()


# ─────────────────────────────────────────────────────────────
# Step 1: Train all variants
# ─────────────────────────────────────────────────────────────

def train_variants(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    """
    Trains variants A, B, C, D.
    Skips a variant if its best.pt already exists.

    Returns:
        dict mapping variant letter → checkpoint directory Path
    """
    ckpt_dirs: dict[str, Path] = {}

    for variant in ["A", "B", "C", "D"]:
        ckpt_dir = root / "checkpoints" / f"variant_{variant}"
        ckpt_dirs[variant] = ckpt_dir

        if checkpoint_exists(ckpt_dir):
            print(f"\n  Variant {variant}: checkpoint exists, skipping training.")
            continue

        cmd = [
            python(), str(root / "train.py"),
            "--variant",    variant,
            "--data_root",  args.data_root,
            "--output_dir", str(ckpt_dir),
            "--epochs",     "2" if args.fast else str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
        ]
        if args.fast:
            cmd.append("--fast_dev_run")

        rc = run_cmd(cmd, f"Train Variant {variant}")
        if rc != 0:
            print(f"  Training Variant {variant} failed. Continuing with other variants.")

    return ckpt_dirs


# ─────────────────────────────────────────────────────────────
# Step 2: Evaluate all variants (ablation)
# ─────────────────────────────────────────────────────────────

def run_ablation_eval(args: argparse.Namespace, root: Path) -> None:
    """Runs eval.py --ablation to compare all 4 variants."""
    out_dir = root / "results" / "ablation"
    done    = out_dir / "ablation_metrics.json"

    if done.exists():
        print(f"\n  Ablation results exist at {done}, skipping.")
        return

    cmd = [
        python(), str(root / "eval.py"),
        "--ablation",
        "--checkpoints_dir", str(root / "checkpoints"),
        "--data_root",       args.data_root,
        "--output_dir",      str(out_dir),
        "--T",               str(args.T),
        "--num_workers",     str(args.num_workers),
    ]
    run_cmd(cmd, "Ablation evaluation (all 4 variants)")


# ─────────────────────────────────────────────────────────────
# Step 3: Full evaluation of Variant D
# ─────────────────────────────────────────────────────────────

def run_eval_variant_d(args: argparse.Namespace, root: Path) -> None:
    """Runs eval.py for Variant D with full figures."""
    ckpt    = root / "checkpoints" / "variant_D" / "best.pt"
    out_dir = root / "results" / "eval_D"
    done    = out_dir / "metrics.json"

    if not ckpt.exists():
        print(f"\n  Variant D checkpoint not found at {ckpt}, skipping eval.")
        return

    if done.exists():
        print(f"\n  Variant D eval results exist, skipping.")
        return

    cmd = [
        python(), str(root / "eval.py"),
        "--checkpoint",  str(ckpt),
        "--data_root",   args.data_root,
        "--output_dir",  str(out_dir),
        "--T",           str(args.T),
        "--n_maps",      str(args.n_maps),
        "--num_workers", str(args.num_workers),
    ]
    run_cmd(cmd, "Evaluate Variant D (Bolivia OOD)")


# ─────────────────────────────────────────────────────────────
# Step 4: MC Dropout uncertainty for Variant D
# ─────────────────────────────────────────────────────────────

def run_uncertainty(args: argparse.Namespace, root: Path) -> None:
    """Runs uncertainty.py (MC Dropout inference + calibration) for Variant D."""
    ckpt    = root / "checkpoints" / "variant_D" / "best.pt"
    out_dir = root / "results" / "uncertainty"
    done    = out_dir / "uncertainty_metrics.json"

    if not ckpt.exists():
        print(f"\n  Variant D checkpoint not found, skipping uncertainty.")
        return

    if done.exists():
        print(f"\n  Uncertainty results exist, skipping.")
        return

    cmd = [
        python(), str(root / "uncertainty.py"),
        "--checkpoint",  str(ckpt),
        "--data_root",   args.data_root,
        "--output_dir",  str(out_dir),
        "--T",           str(args.T),
        "--n_maps",      str(args.n_maps),
        "--num_workers", str(args.num_workers),
    ]
    run_cmd(cmd, "MC Dropout uncertainty + calibration (Variant D)")


# ─────────────────────────────────────────────────────────────
# Step 5: Population exposure for Variant D
# ─────────────────────────────────────────────────────────────

def run_exposure(args: argparse.Namespace, root: Path) -> None:
    """Runs 06_exposure.py for Variant D."""
    ckpt    = root / "checkpoints" / "variant_D" / "best.pt"
    out_dir = root / "results" / "exposure"
    done    = out_dir / "exposure_results.json"

    if not ckpt.exists():
        print(f"\n  Variant D checkpoint not found, skipping exposure.")
        return

    if done.exists():
        print(f"\n  Exposure results exist, skipping.")
        return

    cmd = [
        python(), str(root / "06_exposure.py"),
        "--checkpoint",  str(ckpt),
        "--data_root",   args.data_root,
        "--pop_dir",     args.pop_dir,
        "--output_dir",  str(out_dir),
        "--T",           str(args.T),
        "--num_workers", str(args.num_workers),
    ]
    run_cmd(cmd, "Population exposure analysis (Variant D)")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_summary(root: Path) -> None:
    """Prints a summary of all results found."""
    print(f"\n{'='*60}")
    print("  Experiment Summary")
    print(f"{'='*60}")

    ablation_path = root / "results" / "ablation" / "ablation_metrics.json"
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text())
        print("\n  Ablation Results:")
        print(f"  {'Variant':<10} {'IoU':>8} {'F1':>8} {'ECE':>8} {'Brier':>8}")
        print("  " + "-" * 42)
        for v in ["A", "B", "C", "D"]:
            if v in ablation:
                m = ablation[v]
                print(f"    {v:<8} {m.get('iou', 0):>8.4f} {m.get('f1', 0):>8.4f} "
                      f"{m.get('ece', 0):>8.4f} {m.get('brier', 0):>8.4f}")

    exposure_path = root / "results" / "exposure" / "exposure_results.json"
    if exposure_path.exists():
        exp = json.loads(exposure_path.read_text())
        overall = exp.get("overall", {})
        print(f"\n  Population Exposure (Variant D, Bolivia OOD):")
        print(f"    Gated (trusted)   : {overall.get('gated_exposure', 0):>12,.0f} people")
        print(f"    Deterministic     : {overall.get('deterministic_exposure', 0):>12,.0f} people")
        print(f"    Uncertain zone    : {overall.get('uncertain_exposure', 0):>12,.0f} people")

    uq_path = root / "results" / "uncertainty" / "uncertainty_metrics.json"
    if uq_path.exists():
        uq = json.loads(uq_path.read_text())
        overall = uq.get("overall", {})
        print(f"\n  Calibration (Variant D):")
        print(f"    ECE             : {overall.get('ece', 0):.4f}  (target < 0.05)")
        print(f"    Brier score     : {overall.get('brier', 0):.4f}")
        print(f"    Mean variance   : {overall.get('mean_variance', 0):.4f}")

    print(f"\n  All results in: {root / 'results'}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    root = Path(__file__).parent

    print(f"\n{'='*60}")
    print("  TerrainFlood-UQ — Full Experiment")
    print(f"{'='*60}")
    print(f"  Data root  : {args.data_root}")
    print(f"  Fast mode  : {args.fast}")
    print(f"  Eval only  : {args.eval_only}")
    print(f"{'='*60}\n")

    # Step 1: Train
    if not args.eval_only:
        train_variants(args, root)
    else:
        print("Skipping training (--eval_only).")

    # Step 2: Ablation comparison
    run_ablation_eval(args, root)

    # Step 3: Full eval of Variant D
    run_eval_variant_d(args, root)

    # Step 4: Uncertainty (Variant D only — requires MC Dropout)
    if not args.skip_uncertainty:
        run_uncertainty(args, root)

    # Step 5: Exposure (Variant D only)
    if not args.skip_exposure:
        run_exposure(args, root)

    # Final summary
    print_summary(root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TerrainFlood-UQ End-to-End Experiment")
    p.add_argument("--data_root",    type=str, default="data/sen1floods11")
    p.add_argument("--pop_dir",      type=str, default="data/pop_chips",
                   help="Directory with *_pop.tif files for exposure analysis")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--T",            type=int, default=20,
                   help="MC Dropout forward passes")
    p.add_argument("--n_maps",       type=int, default=10,
                   help="Number of flood map figures to save")
    p.add_argument("--fast",         action="store_true",
                   help="Quick smoke test: 2 epochs with 2 batches each")
    p.add_argument("--eval_only",    action="store_true",
                   help="Skip training; use existing checkpoints")
    p.add_argument("--skip_uncertainty", action="store_true",
                   help="Skip uncertainty.py step")
    p.add_argument("--skip_exposure", action="store_true",
                   help="Skip 06_exposure.py step")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
