# CLAUDE.md — TerrainFlood-UQ Project Rules

Rules for Claude Code in this project. Read this file before touching any code.

---

## Data tensor: 6 bands — Encoder input: 4 SAR bands (variant-dependent)

WorldPop is **NOT** a model input. It is used only in `06_exposure.py` post-prediction
for population exposure estimation.

### Band table — `batch["image"]` channels 0–5

| Index | Name         | Source      | Units    | Used by encoder?                              |
|-------|--------------|-------------|----------|-----------------------------------------------|
| 0     | VV_pre       | Sentinel-1  | dB       | ✓ All variants (pre branch)                   |
| 1     | VH_pre       | Sentinel-1  | dB       | ✓ All variants (pre branch)                   |
| 2     | VV_post      | Sentinel-1  | dB       | ✓ All variants (post branch)                  |
| 3     | VH_post      | Sentinel-1  | dB       | ✓ All variants (post branch)                  |
| 4     | VV_VH_ratio  | Derived     | dB       | ✗ In tensor; NOT fed to encoder (future work) |
| 5     | HAND         | MERIT Hydro | z-score* | ✗ Not to encoder — routed to gate (C/D) or    |
|       |              |             |          |   concatenated to branches (B only)           |

`*` HAND is z-score normalised in the tensor. `_prepare_hand()` in `03_model.py`
**denormalises back to metres** (mean=9.346 m, std=28.330 m) before the gate so
that `exp(−h / 50.0)` operates on the correct physical scale.

### Encoder input per ablation variant

| Variant | Encoder channels (per branch) | HAND usage                  |
|---------|-------------------------------|-----------------------------|
| A       | VV, VH  (2-ch)                | not used                    |
| B       | VV, VH, HAND  (3-ch)          | concatenated to both branches |
| C       | VV, VH  (2-ch)                | routed to HAND gate (metres) |
| D       | VV, VH  (2-ch)                | routed to HAND gate (metres) + MC Dropout |

**WorldPop** (`pop_log`) is loaded by `02_dataset.py` into `batch["pop"]` separately.
It does NOT appear in `batch["image"]` and does NOT enter the model forward pass.

---

## Coding rules

1. **Always use PyTorch**, not TensorFlow.
2. **Never rewrite working files** — only edit what is broken or extend what is needed.
3. **Keep modules single-responsibility** — one file per phase, clean imports.
4. **MC Dropout must be active at inference** — call `model.enable_dropout()` after `model.eval()`. Never use plain `model.eval()` during MC passes.
5. **Bolivia is always test** — never include it in training or validation splits.
6. **All rasterio operations** must handle `nodata` and reproject to EPSG:4326 at 10 m.
7. **Type hints on all functions.** Docstrings on all classes and public methods.
8. **Log everything to TensorBoard** — loss, IoU, ECE per epoch, sample predictions.
9. When in doubt about a design choice, check what the **ablation study requires** — all 4 variants must be runnable from the same codebase via a config flag.
10. **Checkpoints save**: model weights + optimiser state + epoch + best val IoU + config dict.
11. **6-band data tensor, 4-band encoder** — the dataset outputs 6 bands but the encoder uses 4 SAR channels (0–3). Channel 4 (VV_VH_ratio) is precomputed but not currently fed to the encoder. Channel 5 (HAND) is routed to the attention gate after denormalisation to metres. Any reference to "7-band" is outdated.
12. **HAND gate physics** — `_prepare_hand()` must denormalise channel 5 from z-score to metres before passing to `HANDAttentionGate`. Constants: mean=9.346 m, std=28.330 m (from `norm_stats.json`). Never remove this denormalisation step.
13. **Variant B must be retrained** if it was trained before `norm_stats.json` was computed (placeholder HAND mean=5, std=10). Check training log for the WARNING message.

---

## Do NOT touch

- `01_gee_export.py` — GEE exports are complete
- `06_exposure.py` — handles WorldPop post-prediction; pop does NOT go through the model
- `run_experiment.py` — orchestration only
- `jobs/train.sh` — SLURM script for DKUCC

---

## Files and their phases

| File              | Phase | Status     |
|-------------------|-------|------------|
| 01_gee_export.py  | 1     | ✓ Complete |
| 02_dataset.py     | 1     | ✓ Complete |
| 03_model.py       | 2     | ✓ Complete |
| train.py          | 3     | ✓ Complete |
| 05_uncertainty.py | 4     | ✓ Complete |
| 06_exposure.py    | 5     | ✓ Complete |
| eval.py           | 6     | ✓ Complete |
| plots.py          | 6     | ✓ Complete |
| run_experiment.py | —     | ✓ Complete |
