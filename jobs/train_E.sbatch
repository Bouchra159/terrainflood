#!/bin/bash
#SBATCH --job-name=TerrainFlood_E
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_E_%j.out
#SBATCH --error=logs/train_E_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Variant E — True Siamese Change-Detection + HAND gate + MC Dropout
# ─────────────────────────────────────────────────────────────────────────────
# What changed vs Variant D:
#   diff_mode=True in SiameseEncoder.forward()
#   The decoder now sees (post_feats[i] - pre_feats[i]) at every pyramid level
#   instead of post_feats[i] only.
#
#   Architecture parameters: identical to D (same 24.5M params)
#   Checkpoints: NOT compatible with D (different encoder forward path)
#   Training: start from ImageNet-pretrained ResNet-34 (same as D)
#
# Hypothesis: change-detection encoding should improve precision by reducing
# false positives from permanent water bodies that look similar to flood
# in post-event SAR alone.
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

# Variant E: TRUE Siamese difference + HAND gate + MC Dropout
# Slightly longer time limit than D (10h) in case diff encoding slows convergence
python -u train.py \
    --variant E \
    --data_root data/sen1floods11 \
    --output_dir checkpoints/variant_E \
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
