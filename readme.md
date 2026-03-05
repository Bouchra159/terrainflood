# TerrainFlood-UQ — Master's Thesis Project

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
├── uncertainty.py        Phase 4 — MC Dropout inference + calibration          ✓ Complete
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
│   ├── audit_pipeline.py        ← Run before any training to verify dataset + IoU
│   ├── export_tb_curves.py      ← Export TensorBoard curves to CSV/PNG
│   ├── make_figures.py          ← Paper-quality figure generation
│   └── prediction_figures.py    ← Per-chip prediction visualisation
├── jobs/
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

# Evaluate
python eval.py --checkpoint checkpoints/variant_D/best.pt \
               --data_root data/sen1floods11 --T 20
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

## Key papers

1. **Bonafilia et al. (2020)** — Sen1Floods11 dataset. Our benchmark.
2. **Islam et al. (2025)** — DeepSAR Flood Mapper. GEE+HAND+MLP. We beat this with spatial CNN + UQ.
3. **Saleh et al. (2025)** — DeepSARFlood. ViT ensembles. SotA IoU≈0.72. Our comparison target.
4. **Ludwig & Hänsch (2025)** — Post-hoc UQ for SAR flood detection. Validates calibration approach.
5. **Garshasbi et al. (2025)** — Bayesian DL for SAR flood mapping. Closest competitor.
6. **Nobre et al. (2011)** — Original HAND paper. Physics motivation.

---

## Git workflow (VPN constraint)

```
VPN ON  → Claude Code works, SSH to DKUCC blocked
            → write code → commit → push to GitHub

VPN OFF → SSH to DKUCC works, Claude Code blocked
            → git pull → sbatch → disconnect (GPU keeps running)
```

GitHub is the bridge between local development and DKUCC.

---

## Rules for Claude Code

See `CLAUDE.md` for the authoritative rule set. Key points:
- 6-band model input (never 7)
- Bolivia is always test (never train/val)
- `model.enable_dropout()` required after `model.eval()` for MC passes
- Log loss + IoU + LR to TensorBoard every epoch
- All 4 ablation variants must run from the same codebase
