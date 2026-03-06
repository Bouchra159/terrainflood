#!/bin/bash
#SBATCH --job-name=TerrainFlood_B
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_B_%j.out
#SBATCH --error=logs/train_B_%j.err

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

# Variant B: HAND as extra input band — no gate, no MC Dropout.
# HAND is concatenated to each SAR branch (3-channel encoder input).
# NOTE: must be trained with real HAND norm stats (norm_stats.json present)
# for a fair comparison with A/C/D.  Verify norm_stats.json exists before
# submitting this job.
# All hyperparameters are identical across A/B/C/D for a fair ablation.
python -u train.py \
    --variant B \
    --data_root data/sen1floods11 \
    --output_dir checkpoints/variant_B \
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
