#!/bin/bash
#SBATCH --job-name=TerrainFlood_C
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_C_%j.out
#SBATCH --error=logs/train_C_%j.err

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

python -u train.py \
    --variant C \
    --data_root data/sen1floods11 \
    --output_dir checkpoints/variant_C \
    --epochs 50 \
    --batch_size 8 \
    --patch_size 256 \
    --lr 1e-4 \
    --pos_weight 10.0 \
    --loss_alpha 0.5 \
    --early_stopping_patience 10 \
    --num_workers 4

echo "=== JOB DONE ==="
date
