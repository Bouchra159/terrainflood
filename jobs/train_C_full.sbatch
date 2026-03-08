#!/bin/bash
#SBATCH --job-name=TF_C_full
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_C_full_%j.out
#SBATCH --error=logs/train_C_full_%j.err

# Variant C full retraining — HAND gate, no dropout.
# Key changes vs train_C.sbatch:
#   --epochs 120       (was 50; C best-val was at epoch 4, patience fired too early)
#   --early_stopping_patience 25  (was 10; val IoU is noisy on small val set)
#   --time 12:00:00    (was 6h; extra margin for longer convergence)
# Checkpoint saves to checkpoints/variant_C_full/ to avoid overwriting the
# existing best.pt used for the paper figures.

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
    --variant C \
    --data_root data/sen1floods11 \
    --output_dir checkpoints/variant_C_full \
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
