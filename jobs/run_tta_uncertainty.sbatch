#!/bin/bash
#SBATCH --job-name=TerrainFlood_TTA
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_tta_%j.out
#SBATCH --error=logs/run_tta_%j.err

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

# TTA uncertainty estimation on Bolivia test set (Variant D).
# TTA (8 augmentations: 4 rotations x 2 flips) provides 5–50x larger
# variance signal than MC Dropout for overconfident models (T=0.1).
# Also runs temperature calibration on val set before test inference.
#
# Outputs:
#   results/uncertainty_tta/uncertainty_metrics.json
#   results/uncertainty_tta/maps/            (per-chip uncertainty panels)
#   results/uncertainty_tta/arrays/          (mean_prob + variance .npy)
python -u 05_uncertainty.py \
    --checkpoint checkpoints/variant_D/best.pt \
    --data_root  data/sen1floods11 \
    --output_dir results/uncertainty_tta \
    --T 20 \
    --use_tta \
    --calibrate \
    --analyze_logits \
    --save_arrays \
    --n_maps 15 \
    --batch_size 4 \
    --num_workers 4

echo "=== TTA DONE ==="

# Also run standard MC Dropout for comparison
python -u 05_uncertainty.py \
    --checkpoint checkpoints/variant_D/best.pt \
    --data_root  data/sen1floods11 \
    --output_dir results/uncertainty_mc \
    --T 30 \
    --calibrate \
    --analyze_logits \
    --save_arrays \
    --n_maps 15 \
    --batch_size 4 \
    --num_workers 4

echo "=== MC DONE ==="
echo "=== JOB DONE ==="
date
