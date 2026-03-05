"""
Phase 3 — Training Loop
========================
File: train.py

Full PyTorch training loop for the TerrainFlood-UQ project.
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Mixed precision training (torch.cuda.amp)
  - TensorBoard: train/val loss, IoU, LR, sample predictions every 5 epochs
  - Checkpoints: best.pt + latest.pt with full state
  - Early stopping (patience=10)
  - Ablation variants A/B/C/D via --variant flag

Usage:
  python train.py --variant D --data_root data/sen1floods11 --output_dir checkpoints/variant_D
  python train.py --variant A --fast_dev_run   # smoke test (2 batches)
"""

import json
import argparse
import random
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Numeric-prefix module imports
# ─────────────────────────────────────────────────────────────

def _import_module(alias: str, file_path: str):
    """Load a Python file with a numeric prefix as a named module."""
    spec = importlib.util.spec_from_file_location(alias, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_root = Path(__file__).parent
if "model"   not in sys.modules:
    _import_module("model",   str(_root / "03_model.py"))
if "dataset" not in sys.modules:
    _import_module("dataset", str(_root / "02_dataset.py"))

from model   import build_model, FloodLoss          # noqa: E402
from dataset import get_dataloaders                  # noqa: E402


# ─────────────────────────────────────────────────────────────
# 1.  Utilities
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_iou(
    logits:       torch.Tensor,
    labels:       torch.Tensor,
    threshold:    float = 0.5,
    ignore_value: int   = -1,
) -> float:
    """Binary IoU for the flood class; ignores label == ignore_value pixels."""
    probs   = torch.sigmoid(logits.squeeze(1))   # (B, H, W)
    preds   = (probs > threshold).long()
    valid   = labels != ignore_value
    preds   = preds[valid]
    targets = labels[valid]

    intersection = ((preds == 1) & (targets == 1)).sum().item()
    union        = ((preds == 1) | (targets == 1)).sum().item()
    return intersection / (union + 1e-6)


def save_checkpoint(
    out_dir:   Path,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    best_iou:  float,
    config:    dict,
    is_best:   bool,
) -> None:
    """Saves latest.pt always; overwrites best.pt when is_best=True."""
    state = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch":           epoch,
        "best_iou":        best_iou,
        "config":          config,
    }
    torch.save(state, out_dir / "latest.pt")
    if is_best:
        torch.save(state, out_dir / "best.pt")


def load_checkpoint(
    ckpt_path: str,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device:    torch.device = torch.device("cpu"),
) -> dict:
    """Loads checkpoint; returns the state dict."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}  best_iou={ckpt['best_iou']:.4f}")
    return ckpt


# ─────────────────────────────────────────────────────────────
# 2.  TensorBoard sample predictions
# ─────────────────────────────────────────────────────────────

def log_sample_predictions(
    writer:  SummaryWriter,
    model:   torch.nn.Module,
    batch:   dict,
    device:  torch.device,
    epoch:   int,
    tag:     str = "val",
    n:       int = 4,
) -> None:
    """
    Logs up to n sample predictions as image grids to TensorBoard.
    Layout per sample (concatenated along width):
      SAR post-VV  |  flood probability  |  ground truth
    """
    model.eval()
    images = batch["image"][:n].to(device)
    labels = batch["label"][:n]

    with torch.no_grad():
        logits = model(images)
        probs  = torch.sigmoid(logits.squeeze(1)).cpu()   # (N, H, W)

    # SAR VV post (channel index 2) — normalise for display
    vv_post  = images[:, 2:3, :, :].cpu()
    vv_disp  = (vv_post - vv_post.min()) / (vv_post.max() - vv_post.min() + 1e-6)

    pred_disp  = probs.unsqueeze(1)                          # (N, 1, H, W)
    label_disp = (labels > 0).float().unsqueeze(1)           # (N, 1, H, W)

    # Concatenate panels along width dimension for each sample
    grid = torch.cat([vv_disp, pred_disp, label_disp], dim=-1)   # (N, 1, H, 3W)
    writer.add_images(f"{tag}/sar_pred_label", grid, epoch)


# ─────────────────────────────────────────────────────────────
# 3.  Training epoch
# ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     torch.nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
    device:    torch.device,
    grad_clip: float = 1.0,
    fast_dev:  bool  = False,
) -> dict:
    """One training epoch with mixed precision and gradient clipping."""
    model.train()
    total_loss = total_iou = n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for i, batch in enumerate(pbar):
        if fast_dev and i >= 2:
            break

        images = batch["image"].to(device)   # (B, 6, H, W)
        labels = batch["label"].to(device)   # (B, H, W)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)           # (B, 1, H, W)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        iou = compute_iou(logits.detach(), labels.detach())
        total_loss += loss.item()
        total_iou  += iou
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    n = max(n_batches, 1)
    return {"loss": total_loss / n, "iou": total_iou / n}


# ─────────────────────────────────────────────────────────────
# 4.  Validation epoch
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:     torch.nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device:    torch.device,
    fast_dev:  bool = False,
) -> tuple[dict, dict | None]:
    """
    Validation pass.
    Returns (metrics_dict, first_batch) — first_batch for TensorBoard visualisation.
    """
    model.eval()
    total_loss = total_iou = n_batches = 0
    first_batch = None

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for i, batch in enumerate(pbar):
        if fast_dev and i >= 2:
            break
        if first_batch is None:
            first_batch = batch

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss   = criterion(logits, labels)

        iou = compute_iou(logits, labels)
        total_loss += loss.item()
        total_iou  += iou
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    n = max(n_batches, 1)
    return {"loss": total_loss / n, "iou": total_iou / n}, first_batch


# ─────────────────────────────────────────────────────────────
# 5.  Main training loop
# ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "variant":                    args.variant,
        "lr":                         args.lr,
        "weight_decay":               args.weight_decay,
        "epochs":                     args.epochs,
        "batch_size":                 args.batch_size,
        "patch_size":                 args.patch_size,
        "loss_alpha":                 args.loss_alpha,
        "pos_weight":                 args.pos_weight,
        "grad_clip":                  args.grad_clip,
        "early_stopping_patience":    args.early_stopping_patience,
        "seed":                       args.seed,
        "data_root":                  args.data_root,
    }

    print(f"\n{'='*60}")
    print(f"  TerrainFlood-UQ  |  Variant {args.variant}  |  {device}")
    print(f"{'='*60}")
    print(f"  Output  : {out_dir}")
    print(f"  Data    : {args.data_root}")
    print(json.dumps(config, indent=2))
    print(f"{'='*60}\n")

    # ── Data
    train_loader, val_loader, _ = get_dataloaders(
        data_root   = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        patch_size  = args.patch_size,
        pin_memory  = device.type == "cuda",
    )

    # ── Model
    model = build_model(variant=args.variant, pretrained=args.pretrained)
    model = model.to(device)

    # ── Loss
    criterion = FloodLoss(pos_weight=args.pos_weight, alpha=args.loss_alpha)

    # ── Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Resume
    start_epoch = 0
    best_iou    = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt        = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt["best_iou"]
        print(f"  Resuming from epoch {start_epoch}")

    # ── TensorBoard
    tb_dir = out_dir / "runs"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: tensorboard --logdir {tb_dir}\n")

    # ── Save config
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ── Training loop
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Epoch {epoch+1:03d}/{args.epochs:03d}]")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, grad_clip=args.grad_clip, fast_dev=args.fast_dev_run,
        )
        val_metrics, first_val_batch = val_epoch(
            model, val_loader, criterion, device,
            fast_dev=args.fast_dev_run,
        )

        lr_now = optimizer.param_groups[0]["lr"]   # current LR before stepping
        scheduler.step()

        print(f"  Train  loss={train_metrics['loss']:.4f}  iou={train_metrics['iou']:.4f}")
        print(f"  Val    loss={val_metrics['loss']:.4f}    iou={val_metrics['iou']:.4f}")
        print(f"  LR     {lr_now:.2e}")

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_metrics["loss"],
                                    "val":   val_metrics["loss"]},  epoch)
        writer.add_scalars("IoU",  {"train": train_metrics["iou"],
                                    "val":   val_metrics["iou"]},   epoch)
        writer.add_scalar("LR", lr_now, epoch)

        # Sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0 and first_val_batch is not None:
            log_sample_predictions(writer, model, first_val_batch, device, epoch)

        # Checkpoint
        is_best = val_metrics["iou"] > best_iou
        if is_best:
            best_iou         = val_metrics["iou"]
            patience_counter = 0
            print(f"  ✓ New best IoU: {best_iou:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(out_dir, model, optimizer, epoch, best_iou, config, is_best)

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping (patience={args.early_stopping_patience})")
            break

        if args.fast_dev_run:
            print("fast_dev_run: stopping after 1 epoch.")
            break

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Done. Best val IoU: {best_iou:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# 6.  Argument parser
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TerrainFlood-UQ Training")
    p.add_argument("--variant",    type=str,   default="D",
                   choices=["A", "B", "C", "D"],
                   help="Ablation variant (A=SAR only, D=full model)")
    p.add_argument("--data_root",  type=str,   default="data/sen1floods11")
    p.add_argument("--output_dir", type=str,   default="checkpoints/variant_D")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--patch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss_alpha", type=float, default=0.5,
                   help="alpha*BCE + (1-alpha)*Dice")
    p.add_argument("--pos_weight", type=float, default=10.0,
                   help="BCE positive class weight for flood pixels")
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--fast_dev_run", action="store_true",
                   help="Run only 2 batches per epoch (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
