#!/bin/bash
#SBATCH --job-name=TerrainFlood_EvalAll
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/run_eval_all_%j.out
#SBATCH --error=logs/run_eval_all_%j.err

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

echo "=== FULL ABLATION EVALUATION ==="

# 1. Full ablation table (auto-detects all available checkpoints)
python -u eval.py \
    --ablation \
    --checkpoints_dir checkpoints \
    --data_root data/sen1floods11 \
    --output_dir results/ablation \
    --T 20 \
    --batch_size 4 \
    --num_workers 4

echo "=== ABLATION TABLE DONE ==="

# 2. Per-variant individual evaluations (Bolivia OOD)
for VARIANT in A B C D E D_plus baseline_unet; do
    CKPT="checkpoints/variant_${VARIANT}/best.pt"
    if [ -f "$CKPT" ]; then
        echo "--- Evaluating Variant ${VARIANT} ---"
        python -u eval.py \
            --checkpoint "$CKPT" \
            --data_root  data/sen1floods11 \
            --output_dir "results/eval_${VARIANT}" \
            --split test \
            --T 20 \
            --n_maps 15 \
            --batch_size 4 \
            --num_workers 4
    else
        echo "--- Variant ${VARIANT}: checkpoint not found, skipping ---"
    fi
done

echo "=== PER-VARIANT EVAL DONE ==="

# 3. Threshold sweep across all trained variants
python -u tools/threshold_sweep.py \
    --checkpoints_dir checkpoints \
    --data_root data/sen1floods11 \
    --output_dir results/threshold_sweep \
    --split test \
    --num_workers 4

echo "=== THRESHOLD SWEEP DONE ==="

# 4. IoU vs HAND analysis
python -u tools/iou_vs_hand.py \
    --checkpoints_dir checkpoints \
    --data_root data/sen1floods11 \
    --output_dir results/iou_vs_hand \
    --split test \
    --num_workers 4

echo "=== IOUvsHAND DONE ==="

# 5. Gate attention α vs HAND (Variants C, D, E, D_plus only)
python -u tools/alpha_vs_hand.py \
    --checkpoints_dir checkpoints \
    --data_root data/sen1floods11 \
    --output_dir results/alpha_vs_hand \
    --split test \
    --n_chips 15 \
    --scatter \
    --num_workers 4

echo "=== ALPHA_VS_HAND DONE ==="

# 6. Bootstrap CIs on ablation table
python -u tools/bootstrap_ci.py \
    --checkpoints_dir checkpoints \
    --data_root data/sen1floods11 \
    --output_dir results/bootstrap_ci \
    --n_bootstrap 1000 \
    --split test \
    --num_workers 4

echo "=== BOOTSTRAP CI DONE ==="

echo "=== ALL EVALUATION COMPLETE ==="
date
