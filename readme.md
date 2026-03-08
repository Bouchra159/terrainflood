**Physics-informed, uncertainty-aware deep learning for flood mapping from SAR satellite imagery.**

---

## Research contribution

Three things combined that no existing paper does together:

1. **HAND attention gate** — Height Above Nearest Drainage as a physics prior inside the decoder attention mechanism. Low HAND (near river) → gate opens → flood predictions allowed. High HAND (hillside) → gate closes → false positives suppressed. Not just an extra input band — it controls information flow in the network.

2. **MC Dropout uncertainty** — T=20 stochastic forward passes at inference. Predictive variance → trust mask. Only confident pixels count toward metrics and downstream analysis.

3. **Uncertainty-gated population exposure** — WorldPop × flood probability, summed only inside the trust mask. Output: "X people confidently affected, Y people in uncertain zones."

---

## Tech stack

- **Python 3.11**, **PyTorch** (not TensorFlow)
- **Google Earth Engine** Python API (earthengine-api)
- **Rasterio** for GeoTIFF I/O (all ops at EPSG:4326, 10 m)
- **DKUCC cluster** (SLURM, NVIDIA L20) for GPU training
- **TensorBoard** for training monitoring

---

## Project structure

```
terrainflood/
├── 01_gee_export.py      Phase 1 — GEE: export S1 + HAND + WorldPop chips    ✓ Complete
├── 02_dataset.py         Phase 1 — PyTorch Dataset + DataLoader factory        ✓ Complete
├── 03_model.py           Phase 2 — Architecture (Siamese + HAND gate)          ✓ Complete
├── train.py              Phase 3 — Training loop + AMP + TensorBoard           ✓ Complete
├── 05_uncertainty.py     Phase 4 — MC Dropout inference + calibration          ✓ Complete
├── 06_exposure.py        Phase 5 — Population exposure + confidence bounds     ✓ Complete
├── eval.py               Phase 6 — Metrics, ablation table                     ✓ Complete
├── plots.py              Phase 6 — All figure generation                       ✓ Complete
├── trust_mask.py                — Trust mask utilities (used by eval + exposure)
├── run_experiment.py            — End-to-end pipeline orchestration (DO NOT TOUCH)
├── export_watchdog.py           — GEE export monitoring
├── config.yaml                  — Reference configuration (argparse takes precedence)
├── environment.yml              — Conda environment definition
├── requirements.txt             — pip-installable subset
├── tools/
│   ├── audit_pipeline.py              ← Run before any training to verify dataset + IoU
│   ├── export_tb_curves.py            ← Export TensorBoard scalars to CSV/PNG
│   ├── make_figures.py                ← Dataset/paper-quality figure generation
│   ├── plot_training_curves_overlay.py← Multi-variant training curve overlay figure
│   ├── prediction_figures.py          ← Per-chip prediction visualisation
│   └── visualize_gate.py              ← HAND attention gate map visualisation (C/D)
├── jobs/
│   ├── train_A.sbatch           ← SLURM: train Variant A (SAR baseline)
│   ├── train_B.sbatch           ← SLURM: train Variant B
│   ├── train_C.sbatch           ← SLURM: train Variant C
│   └── train_D.sbatch           ← SLURM: train Variant D (full model)
└── data/
    └── sen1floods11/
        ├── flood_events/        ← gsutil download (HandLabeled chips)
        ├── hand_chips/          ← GEE exports (optional: per-chip HAND .tif)
        └── pop_chips/           ← GEE exports (optional: per-chip WorldPop .tif)
```

---

## Model input: 6 bands (NOT 7)

WorldPop (`pop_log`) is **not** a model input — it is used only in `06_exposure.py` post-prediction for population exposure estimation.

| Index | Name         | Source      | Units   | Physical meaning                  |
|-------|--------------|-------------|---------|-----------------------------------|
| 0     | VV_pre       | Sentinel-1  | dB      | Pre-flood VV backscatter          |
| 1     | VH_pre       | Sentinel-1  | dB      | Pre-flood VH backscatter          |
| 2     | VV_post      | Sentinel-1  | dB      | Post-flood VV backscatter         |
| 3     | VH_post      | Sentinel-1  | dB      | Post-flood VH backscatter         |
| 4     | VV_VH_ratio  | Derived     | dB      | VV − VH (surface roughness proxy) |
| 5     | HAND         | MERIT Hydro | metres  | Height above nearest drainage     |

### Labels (Sen1Floods11 HandLabeled)

| Value | Meaning                                       |
|-------|-----------------------------------------------|
| 0     | No flood                                      |
| 1     | Flood (water)                                 |
| 2     | Permanent water (treated as flood by default) |
| −1    | Invalid / ignore mask                         |

### Data splits

| Split | Events                                                      |
|-------|-------------------------------------------------------------|
| Train | Cambodia, Canada, DemRepCongo, Ghana, India, Mekong, Nigeria, Somalia |
| Val   | Ecuador, Paraguay                                           |
| Test  | **Bolivia** — out-of-distribution holdout, never in train/val |

---

## Ablation variants

| Variant | SAR encoder | HAND | HAND gate | MC Dropout |
|---------|-------------|------|-----------|------------|
| A       | ✓           | ✗    | ✗         | ✗          | ← SAR baseline
| B       | ✓           | band | ✗         | ✗          | ← HAND as extra input band
| C       | ✓           | gate | ✓         | ✗          | ← physics gate, no UQ
| **D**   | ✓           | gate | ✓         | ✓          | ← **full model (paper)**

All variants run from the same codebase via `--variant {A,B,C,D}`.

---

## Model architecture

```
Input: (B, 6, H, W)
  Channels: VV_pre | VH_pre | VV_post | VH_post | VV_VH_ratio | HAND
       │
  ┌────┴────┐
  │  Split  │
  ↓         ↓
[B,2,H,W]  [B,2,H,W]
 pre SAR    post SAR
  │              │
  └──── Siamese ResNet-34 (weight-shared) ────┘
           Feature difference: F_post - F_pre
                    │
         Decoder with HAND attention gates:
           α(x,y) = sigmoid(W_g·g + W_x·skip + W_h·HAND)
           gated_skip = α ⊙ skip
                    │
            Output head → flood logits (B, 1, H, W)
                    │
          MC Dropout (active at train + inference)
```

**Loss:** `0.5 × BCE(pos_weight=10) + 0.5 × Dice`

---

## Training

```bash
# On DKUCC — run audit first
cd ~/terrainflood
python tools/audit_pipeline.py --data_root data/sen1floods11

# Train a variant
sbatch jobs/train_B.sbatch   # Variant B (already complete)
sbatch jobs/train_C.sbatch   # Variant C
sbatch jobs/train_D.sbatch   # Variant D (full model)

# Monitor
squeue -u $USER
tensorboard --logdir checkpoints/variant_D/runs

# Evaluate individual variant (T=20 MC passes for D; T=1 for A/B/C)
python eval.py --checkpoint checkpoints/variant_D/best.pt \
               --data_root data/sen1floods11 --output_dir results/eval_D --T 20

# Evaluate all 4 variants (ablation table)
python eval.py --ablation \
               --checkpoints_dir checkpoints \
               --data_root data/sen1floods11 \
               --output_dir results/ablation

# Gate visualisation (Variants C and D only)
python tools/visualize_gate.py \
    --checkpoint checkpoints/variant_D/best.pt \
    --data_root data/sen1floods11 \
    --output_dir results/gate_maps_D --split test --n_chips 15

# Uncertainty calibration (Variant D only)
python 05_uncertainty.py \
    --checkpoint checkpoints/variant_D/best.pt \
    --data_root data/sen1floods11 \
    --output_dir results/uncertainty_D --T 20

# Training curves
python tools/export_tb_curves.py \
    --logdir checkpoints/variant_D/runs \
    --out_dir results --variant D
python tools/plot_training_curves_overlay.py \
    --curves_dir results/curves --out_dir results/paper_figures
```

---

## Data pipeline

GEE export is a one-time local operation. The dataset (`sen1floods11`) is downloaded directly from the public GCS bucket.

```
LAPTOP (VPN ON, one time only)
  └── python 01_gee_export.py --project PROJECT --drive_folder flood_chips
        └── GEE exports hand_chips/, pop_chips/ → Google Drive

DKUCC (VPN OFF, school WiFi)
  └── gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events data/sen1floods11/
  └── rclone copy gdrive:flood_chips data/  (optional: HAND + pop per chip)
```

---

## Results — Bolivia OOD test set (15 chips)

All metrics on the held-out Bolivia event (out-of-distribution, never seen during training or
validation).  Class balance: flood = 10.1 %, background = 89.9 %.

### Ablation comparison (pre-calibration — fair cross-variant comparison)

| Variant | IoU (95% CI) | F1 | Precision | Recall | Accuracy | ECE↓ | Brier↓ |
|---------|--------------|----|-----------| -------|----------|------|--------|
| A — SAR baseline | 0.408 [0.211, 0.578] | 0.580 | 0.413 | **0.973** | 0.776 | 0.402 | 0.231 |
| B — +HAND band† | 0.441 [0.199, 0.658] | 0.612 | 0.453 | 0.944 | 0.810 | **0.240** | **0.139** |
| C — +HAND gate† | 0.662 [0.450, 0.779] | 0.797 | 0.789 | 0.805 | 0.935 | 0.370 | 0.194 |
| **D — +MC Dropout**‡ | **0.690** [0.459, 0.772] | **0.817** | **0.788** | 0.848 | **0.940** | 0.362 | 0.194 |

95% CI from 1 000 chip-level bootstrap resamples.

† A→B and B→C improvements are **highly significant** (McNemar pixel-level test, p < 0.001).

‡ C→D: IoU improves by +2.8 pts in the main eval run, but the improvement is **statistically
inconclusive**. McNemar slightly favours C at pixel level (Δ = +4 318 pixels for C); the
bootstrap CIs [0.459, 0.772] vs [0.450, 0.779] fully overlap; and MC Dropout stochasticity
(T = 20 passes) introduces ~2 pt run-to-run IoU variability for Variant D.
**D's principal contribution is calibrated uncertainty, not raw IoU gain.**

### Variant D calibration (temperature scaling, T ≈ 0.100)

| | ECE↓ | Brier↓ |
|-|------|--------|
| Pre-calibration (raw model output) | 0.362 | 0.194 |
| **Post-calibration** (T ≈ 0.100) | **0.077** | **0.053** |
| Reduction | **78.6 %** | **72.7 %** |

The raw model output is bimodal — predictions cluster in [0.32, 0.74] before temperature
scaling (model under-confident).  T ≈ 0.100 sharpens logits, filling all 15 calibration bins
and achieving near-diagonal reliability.

> **For the paper:** use pre-cal ECE in the ablation table (fair comparison across all 4
> variants); headline Variant D with post-cal ECE = **0.077** in the calibration section.

### Key findings

- **B→C jump (+22 IoU pts):** Physics-informed gating vastly outperforms naive HAND band
  inclusion.  How HAND is integrated matters as much as whether it is used at all.
- **C→D gain (marginal, +2.8 pts):** Driven by recall (+4.4 %), precision unchanged.
  Within MC Dropout run-to-run variance — do not over-claim IoU improvement.
- **Calibration:** Temperature scaling reduces Variant D ECE by 79 % (0.362 → 0.077).
  Pre-calibration model is systematically under-confident (bimodal mid-range outputs).
- **HAND gate physics:** Gate α correlates with HAND elevation — low-HAND (flood-prone)
  chips receive highest activation; high-HAND chips are suppressed.
- **Risk-coverage (AURC = 0.517):** Curve rises steeply at low coverage due to 10:1 class
  imbalance — at coverage < 0.30 almost all retained pixels are background (var ≈ 0),
  giving IoU ≈ 0.  Expected behaviour; not a model failure.

Training converges fast: C and D reach best validation IoU in ~15 epochs.  A requires 50.

---

## Key papers

1. **Bonafilia et al. (2020)** — Sen1Floods11 dataset. Our benchmark.
2. **Islam et al. (2025)** — DeepSAR Flood Mapper. GEE+HAND+MLP. We beat this with spatial CNN + UQ.
3. **Saleh et al. (2025)** — DeepSARFlood. ViT ensembles. SotA IoU≈0.72. Our comparison target.
4. **Ludwig & Hänsch (2025)** — Post-hoc UQ for SAR flood detection. Validates calibration approach.
5. **Garshasbi et al. (2025)** — Bayesian DL for SAR flood mapping. Closest competitor.
6. **Nobre et al. (2011)** — Original HAND paper. Physics motivation.

---
