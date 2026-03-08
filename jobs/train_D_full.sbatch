#!/bin/bash
#SBATCH --job-name=TF_D_full
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=logs/train_D_full_%j.out
#SBATCH --error=logs/train_D_full_%j.err

# Variant D full retraining — HAND gate + MC Dropout.
# Key changes vs train_D.sbatch:
#   --epochs 120       (was 50; D best-val was at epoch 2, patience fired at epoch 12)
#   --early_stopping_patience 25  (was 10)
#   --time 14:00:00    (was 8h; MC Dropout T=20 passes per batch adds ~30% overhead)
# Checkpoint saves to checkpoints/variant_D_full/ to preserve existing best.pt.

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

python -u train.py \
    --variant D \
    --data_root data/sen1floods11 \
    --output_dir checkpoints/variant_D_full \
    --epochs 120 \
    --batch_size 8 \
    --patch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --pos_weight 10.0 \
    --loss_alpha 0.5 \
    --grad_clip 1.0 \
    --early_stopping_patience 25 \
    --num_workers 4 \
    --seed 42

echo "=== JOB DONE ==="
date
