#!/bin/bash
#SBATCH --job-name=TerrainFlood_Dp
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_D_prime_%j.out
#SBATCH --error=logs/train_D_prime_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Variant D-prime — HAND gate + MC Dropout, dropout_rate=0.5
# ─────────────────────────────────────────────────────────────────────────────
# Purpose: fix the tiny-variance problem in Variant D.
#
# Root cause: Variant D uses dropout_rate=0.3 in the decoder only.
# With a well-trained model this produces mean_variance ≈ 3.7e-4 (max ≈ 0.0015),
# so every pixel falls below any sensible tau threshold.  The uncertainty gate
# never fires and the exposure estimate collapses to the deterministic result.
#
# Fix: increase dropout_rate to 0.5.  Higher stochasticity → larger per-pixel
# variance → a meaningful tau sweep becomes possible.
#
# Architecture: identical to D (same HAND gate, same decoder structure).
# The only change is the Dropout layer probability.
# Checkpoint is NOT compatible with D (different dropout weights).
# Uses --dropout_rate 0.5 CLI override — variant letter stays "D" so that
# all downstream eval/uncertainty scripts work unchanged; the override is stored
# in the checkpoint config so the model can be reconstructed exactly.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
echo "=== JOB START ==="
hostname
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PWD=$(pwd)"

cd /dkucc/home/bd159/terrainflood

module purge
module load anaconda/2023.7
module load cuda/12.6

# Try to activate conda env if available
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate terrainflood
fi

mkdir -p logs

# Run audit before training
echo "=== PIPELINE AUDIT ==="
python tools/audit_pipeline.py --data_root data/sen1floods11 --skip_val_loop --skip_sampler
echo "=== AUDIT PASSED — STARTING TRAINING ==="

# Variant D-prime: same as D but dropout_rate=0.5
python -u train.py \
    --variant D \
    --dropout_rate 0.5 \
    --data_root data/sen1floods11 \
    --output_dir checkpoints/variant_D_prime \
    --epochs 50 \
    --batch_size 8 \
    --patch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --pos_weight 10.0 \
    --loss_alpha 0.5 \
    --grad_clip 1.0 \
    --early_stopping_patience 10 \
    --num_workers 4 \
    --seed 42

echo "=== JOB DONE ==="
date
