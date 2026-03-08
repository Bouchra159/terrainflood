#!/bin/bash
#SBATCH --job-name=TF_TTA
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_tta_%j.out
#SBATCH --error=logs/run_tta_%j.err

# TTA + MC Dropout uncertainty on Variant D (Bolivia OOD test set).
#
# Why TTA matters:
#   MC Dropout gives mean_variance ≈ 4e-4 (too small for exposure gating).
#   TTA (8 D4 augmentations) produces 5–50× larger variance signal, which
#   allows the uncertainty threshold in 06_exposure.py to actually gate pixels.
#
# Outputs:
#   results/uncertainty_tta/   — TTA uncertainty (mean_prob, variance, maps)
#   results/uncertainty_mc/    — MC Dropout baseline for comparison (T=30)
#
# Run this BEFORE run_eval_all so exposure can use TTA arrays.

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

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate terrainflood
fi

mkdir -p logs

# ── TTA uncertainty (primary) ───────────────────────────────
python -u 05_uncertainty.py \
    --checkpoint checkpoints/variant_D/best.pt \
    --data_root  data/sen1floods11 \
    --output_dir results/uncertainty_tta \
    --use_tta \
    --n_aug 8 \
    --calibrate \
    --analyze_logits \
    --uncertainty_threshold 0.01 \
    --save_arrays \
    --n_maps 15 \
    --batch_size 4 \
    --num_workers 4

echo "=== TTA DONE ==="

# ── MC Dropout (T=30) for comparison ───────────────────────
python -u 05_uncertainty.py \
    --checkpoint checkpoints/variant_D/best.pt \
    --data_root  data/sen1floods11 \
    --output_dir results/uncertainty_mc \
    --T 30 \
    --calibrate \
    --analyze_logits \
    --uncertainty_threshold 0.001 \
    --save_arrays \
    --n_maps 15 \
    --batch_size 4 \
    --num_workers 4

echo "=== MC DONE ==="
echo "=== JOB DONE ==="
date
