#!/bin/bash
#SBATCH --job-name=TerrainFlood_B
#SBATCH --partition=h20-gpu
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

# Try to activate conda env if available
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate floodgnn2
fi

python -u run_experiment.py --mode B
