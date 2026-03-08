#!/bin/bash
#SBATCH --job-name=TF_EvalAll
#SBATCH --partition=l20-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/run_eval_all_%j.out
#SBATCH --error=logs/run_eval_all_%j.err

# Full evaluation pipeline across all trained variants.
# Evaluates whichever checkpoints exist — skips missing ones gracefully.
#
# Run order on DKUCC:
#   1. sbatch jobs/run_tta_uncertainty.sbatch   (TTA + MC Dropout uncertainty)
#   2. sbatch jobs/run_eval_all.sbatch          (this script — after TTA finishes)

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

# ── 1. Full ablation table (A–D + any extra variants present) ──
echo "=== ABLATION TABLE ==="
python -u eval.py \
    --ablation \
    --checkpoints_dir checkpoints \
    --data_root data/sen1floods11 \
    --output_dir results/ablation \
    --T 20 \
    --batch_size 4 \
    --num_workers 4

# ── 2. Per-variant individual evaluations (Bolivia OOD) ────────
echo "=== PER-VARIANT EVAL ==="
for VARIANT in A B C D E D_plus D_full C_full baseline_unet; do
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
        echo "--- Variant ${VARIANT}: no checkpoint at ${CKPT}, skipping ---"
    fi
done

# ── 3. Exposure analysis with TTA uncertainty ───────────────────
# Uses results/uncertainty_tta/ arrays produced by run_tta_uncertainty.sbatch
echo "=== EXPOSURE ANALYSIS ==="

# Exposure analysis for key models
for VARIANT in D D_full C_full; do
    CKPT="checkpoints/variant_${VARIANT}/best.pt"
    if [ -f "$CKPT" ]; then
        echo "--- Exposure for Variant ${VARIANT} ---"
        python -u 06_exposure.py \
            --checkpoint "$CKPT" \
            --data_root data/sen1floods11 \
            --output_dir "results/exposure_${VARIANT}" \
            --T 20 \
            --batch_size 4 \
            --num_workers 4
    else
        echo "--- Exposure for Variant ${VARIANT}: checkpoint not found, skipping ---"
    fi
done

echo "=== EXPOSURE ANALYSIS DONE ==="
if [ -d "results/uncertainty_tta" ]; then
    python -u 06_exposure.py \
        --uncertainty_dir results/uncertainty_tta \
        --data_root       data/sen1floods11 \
        --output_dir      results/exposure_D_tta \
        --uncertainty_threshold 0.01
else
    echo "No TTA uncertainty dir found — skipping exposure analysis"
fi

echo "=== ALL DONE ==="
date
