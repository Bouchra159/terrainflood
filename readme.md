# FloodMapping — Master's Thesis Project
## Claude Code Project Context

---

## What this project is

A bachelor's graduation project in Earth Observation + Applied Machine Learning.
We are building a **physics-informed, uncertainty-aware deep learning system**
that maps floods from satellite radar imagery and estimates how many people
are affected — with honest confidence bounds.

**One sentence:** SAR satellite sees a flood → our model maps it → tells you
where it's confident → estimates affected population with uncertainty bounds.

---

## The research contribution (what makes this novel)

Three things combined that no existing paper does together:

1. **HAND attention gate** — Height Above Nearest Drainage as a physics prior
   inside the decoder attention mechanism. Low HAND = near river = allow flood.
   High HAND = on a hillside = suppress prediction. Not just an extra input band —
   it controls information flow in the network.

2. **MC Dropout uncertainty** — T=20 stochastic forward passes at inference.
   Predictive variance → trust mask. Only confident pixels count.

3. **Uncertainty-gated population exposure** — WorldPop × flood probability,
   summed only inside the trust mask. Output: "X people definitely affected,
   Y people in uncertain zones." No existing GEE tool does this.

---

## Tech stack

- **Python 3.11**, **PyTorch** (not TensorFlow)
- **Google Earth Engine** Python API (earthengine-api + geemap)
- **Rasterio** for GeoTIFF I/O
- **DKUCC cluster** for training (GPU nodes via SLURM)
- **Claude Code** in VS Code for development

---

## Project structure

```
floodmapping/
├── readme                     ← you are here (project context for Claude Code)
├── pipeline/
│   ├── 01_gee_export.py       ← GEE: export S1 + HAND + WorldPop chips   ✓ DONE
│   ├── 02_dataset.py          ← PyTorch Dataset + DataLoader factory      ✓ DONE
│   ├── 03_model.py            ← Architecture (Siamese + HAND gate)        ← IN PROGRESS
│   ├── 04_train.py            ← Training loop + loss + scheduler
│   ├── 05_uncertainty.py      ← MC Dropout inference + calibration
│   ├── 06_exposure.py         ← Population exposure + confidence bounds
│   └── 07_evaluate.py         ← Metrics, ablations, plots
├── data/
│   └── sen1floods11/
│       ├── flood_events/      ← gsutil download: gs://sen1floods11/v1.1/data/flood_events
│       ├── hand_chips/        ← GEE exports (HAND per chip)
│       └── pop_chips/         ← GEE exports (WorldPop per chip)
├── checkpoints/               ← saved model weights (.pt files)
├── results/                   ← metrics JSON, plots, flood maps
├── notebooks/                 ← exploration, visualisation
├── environment.yml
└── README.md
```

---

## Data

### Input bands (7 channels per chip)
| Index | Name         | Source      | Units       | Physical meaning                     |
|-------|--------------|-------------|-------------|--------------------------------------|
| 0     | VV_pre       | Sentinel-1  | dB          | Pre-flood VV backscatter             |
| 1     | VH_pre       | Sentinel-1  | dB          | Pre-flood VH backscatter             |
| 2     | VV_post      | Sentinel-1  | dB          | Post-flood VV backscatter            |
| 3     | VH_post      | Sentinel-1  | dB          | Post-flood VH backscatter            |
| 4     | VV_VH_ratio  | Derived     | dB          | VV - VH (surface roughness proxy)    |
| 5     | HAND         | MERIT Hydro | metres      | Height above nearest drainage        |
| 6     | pop_log      | WorldPop    | log(pp/km²) | log1p population density             |

### Labels (Sen1Floods11)
- 0 = no flood
- 1 = flood
- 2 = permanent water (treated as flood)
- -1 = invalid / ignore

### Splits
- **Train**: Cambodia, Canada, DemRepCongo, Ghana, India, Mekong, Nigeria, Somalia
- **Val**:   Ecuador, Paraguay
- **Test**:  Bolivia ← OUT-OF-DISTRIBUTION holdout, never touches training

---

## Data pipeline (how chips land on DKUCC)

GEE is a one-time export tool. It runs on your laptop once, puts chips in
Google Drive, and DKUCC downloads them. DKUCC never talks to GEE directly.

```
LAPTOP (VPN ON — one time only)
  └── python 01_gee_export.py --project YOUR_PROJECT --drive_folder flood_chips
        └── GEE queues async export jobs → chips land in Google Drive (~30–60 min)
              Check progress: https://code.earthengine.google.com/tasks

GOOGLE DRIVE
  └── holds flood_chips/ (TFRecord or GeoTIFF, all 11 events)

DKUCC (VPN OFF — one time only, after GEE exports complete)
  └── rclone copy gdrive:flood_chips data/gee_chips/ --progress
  └── gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events data/sen1floods11/
        └── training runs on GPU — no internet needed from here on
```

### Step-by-step: export chips (laptop, VPN ON)

```bash
conda activate terrainflood
# Export one split at a time to stay within Google Drive quota
python 01_gee_export.py --project YOUR_PROJECT --drive_folder flood_chips_train --split train --monitor
python 01_gee_export.py --project YOUR_PROJECT --drive_folder flood_chips_val   --split val   --monitor
python 01_gee_export.py --project YOUR_PROJECT --drive_folder flood_chips_test  --split test  --monitor
```

### Step-by-step: pull chips to DKUCC (SSH, VPN OFF)

```bash
# 1. Authenticate Google Drive once (opens browser link, paste auth code)
rclone config   # New remote → name: gdrive → type: drive → follow prompts

# 2. Download chips from Drive to DKUCC
rclone copy gdrive:flood_chips_train /work/bd159/data/gee_chips/ --progress
rclone copy gdrive:flood_chips_val   /work/bd159/data/gee_chips/ --progress
rclone copy gdrive:flood_chips_test  /work/bd159/data/gee_chips/ --progress

# 3. Download Sen1Floods11 labels (public GCS bucket)
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events \
    /work/bd159/data/sen1floods11/
```

---

## Model architecture (03_model.py)

```
Input: (B, 7, H, W)
         │
    ┌────┴────┐
    │  Split  │
    ↓         ↓
[B,4,H,W]  [B,1,H,W]  [B,1,H,W]
SAR bands   HAND       pop_log
    │
    ↓
Siamese ResNet-34 encoder (weight-shared)
  pre=[VV_pre, VH_pre]   → feature pyramid F_pre
  post=[VV_post,VH_post] → feature pyramid F_post
    │
    ↓
Feature difference: F_post - F_pre  (per scale)
    │
    ↓
Decoder with HAND-gated attention at each skip connection:
  α(x,y) = sigmoid( W_h * HAND(x,y) + W_f * F(x,y) )
  gated_feature = α * F
    │
    ↓
Output head → flood probability (B, 1, H, W)  [0, 1]
    │
MC Dropout (enabled at BOTH train and inference)
```

---

## Training details

- **Loss**: 0.5 × BCE(pos_weight=class_ratio) + 0.5 × DiceLoss
- **Optimiser**: AdamW, lr=1e-4, weight_decay=1e-4
- **Scheduler**: CosineAnnealingLR
- **Batch size**: 8 (with 256×256 random crops)
- **Epochs**: 50 with early stopping (patience=10)
- **Dropout rate**: 0.3 (in decoder, active at train + inference)
- **MC samples at inference**: T=20

---

## Evaluation plan

### Flood mapping metrics (per event + overall)
- IoU (Intersection over Union) for flood class
- F1 score, Precision, Recall
- Stratified by: HAND bin, land cover class, continent

### Uncertainty + calibration
- Expected Calibration Error (ECE)
- Brier score
- Reliability diagrams
- Coverage-accuracy curves

### Population exposure
- Expected exposed population per event
- 90% confidence interval (5th–95th MC percentile)
- Comparison: deterministic vs uncertainty-gated estimate

### Ablation study (4 model variants)
| Variant | SAR | HAND | Gate | UQ |
|---------|-----|------|------|----|
| A       | ✓   |  ✗   |  ✗   | ✗  | ← baseline
| B       | ✓   |  ✓   |  ✗   | ✗  | ← HAND as extra band
| C       | ✓   |  ✓   |  ✓   | ✗  | ← HAND gate, no UQ
| D (ours)| ✓   |  ✓   |  ✓   | ✓  | ← full model

---

## Key papers to know

1. **Bonafilia et al. (2020)** — Sen1Floods11 dataset paper. This is our benchmark.
2. **Islam et al. (2025)** — DeepSAR Flood Mapper. GEE+HAND+MLP. Direct baseline — we beat this by using spatial CNN + UQ.
3. **Saleh et al. (2025)** — DeepSARFlood. ViT ensembles. SotA IoU=0.72 on Sen1Floods11. We compare against this.
4. **Ludwig & Hänsch (2025)** — Post-hoc UQ for SAR flood detection. Validates our calibration approach.
5. **Garshasbi et al. (2025)** — Bayesian DL for SAR flood mapping. Closest competitor. Read before writing novelty claims.
6. **Nobre et al. (2011)** — Original HAND paper. Cite for physics motivation.

---

## Git + GitHub workflow

**Two separate work modes — cannot run both at the same time.**

```
┌──────────────────────────────────────────────────┐
│  MODE A — Duke VPN ON                            │
│  ✓ Claude Code works (needs Anthropic API)       │
│  ✗ SSH to DKUCC blocked                          │
│  → Write code, edit files, commit, push          │
└──────────────────────────────────────────────────┘
                       │
                   GitHub repo
                (single source of truth)
                       │
┌──────────────────────────────────────────────────┐
│  MODE B — VPN OFF (school WiFi only)             │
│  ✗ Claude Code blocked                           │
│  ✓ SSH to DKUCC works                            │
│  → git pull → submit SLURM job → disconnect      │
│  → GPU keeps running after SSH disconnect        │
└──────────────────────────────────────────────────┘
```

**VPN ON session (writing code with Claude Code):**
1. Open VS Code + Claude Code
2. Write/edit the current phase file
3. `git add . && git commit -m "phase X: description"`
4. `git push` — done, switch off VPN

**VPN OFF session (running code on DKUCC):**
1. SSH into DKUCC on school WiFi
2. `git pull`
3. `sbatch train.sh` — GPU job starts, runs independently
4. Disconnect SSH safely — job keeps running
5. Come back later: `squeue -u $USER` to check status

GitHub is the bridge. Never try to do both modes at the same time.

### Disable Claude co-author in commits

Claude Code adds "Co-authored-by: Claude" to commits by default. This is disabled.
The setting lives at `~/.claude/settings.json`:

```json
{
  "includeCoAuthoredBy": false
}
```

Create or edit that file once on your local machine and it applies to all projects permanently. Claude Code will never add co-author attributions to your commits.

---

## Rules for Claude Code in this project

1. **Always use PyTorch**, not TensorFlow, unless a file explicitly says TFRecord I/O.
2. **Never rewrite working files** — only edit what's broken or extend what's needed.
3. **Keep modules single-responsibility** — one file per phase, clean imports.
4. **MC Dropout must be active at inference** — `model.train()` mode during MC passes or use a custom enable_dropout() function. Never `model.eval()` during uncertainty estimation.
5. **Bolivia is always test** — never include it in training or validation splits under any circumstances.
6. **All rasterio operations** must handle `nodata` values and reproject to EPSG:4326 at 10m.
7. **Type hints on all functions**. Docstrings on all classes and public methods.
8. **Log everything to TensorBoard** — loss, IoU, ECE per epoch, sample predictions.
9. When in doubt about a design choice, check what the **ablation study requires** — all 4 variants must be runnable from the same codebase via a config flag.
10. **Checkpoints save**: model weights + optimiser state + epoch + best val IoU + config dict.

---

## Current status

| Phase | File               | Status      |
|-------|--------------------|-------------|
| 1     | 01_gee_export.py   | ✓ Complete  |
| 1     | 02_dataset.py      | ✓ Complete  |
| 2     | 03_model.py        | ✓ Complete  |
| 3     | 04_train.py        | ← Next      |
| 4     | 05_uncertainty.py  | Pending     |
| 5     | 06_exposure.py     | Pending     |
| 6     | 07_evaluate.py     | Pending     |

---

## How to ask Claude Code for help

Good prompts:
- "implement 04_train.py following the training details in readme"
- "add a reliability diagram to 07_evaluate.py"
- "the HAND gate in 03_model.py is producing NaNs, debug it"
- "write a SLURM job script for training on DKUCC with 1 GPU"

Bad prompts:
- "build the flood model" (too vague — always reference the phase number)
- "make it better" (say what metric or behaviour to improve)
