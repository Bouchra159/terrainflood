# CLAUDE.md — TerrainFlood-UQ Project Rules

Rules for Claude Code in this project. Read this file before touching any code.

---

## Model input: 6 bands (NOT 7)

WorldPop is **NOT** a model input. It is used only in `06_exposure.py` post-prediction
for population exposure estimation.

### Input band table (channels 0–5)

| Index | Name         | Source      | Units   | Physical meaning                  |
|-------|--------------|-------------|---------|-----------------------------------|
| 0     | VV_pre       | Sentinel-1  | dB      | Pre-flood VV backscatter          |
| 1     | VH_pre       | Sentinel-1  | dB      | Pre-flood VH backscatter          |
| 2     | VV_post      | Sentinel-1  | dB      | Post-flood VV backscatter         |
| 3     | VH_post      | Sentinel-1  | dB      | Post-flood VH backscatter         |
| 4     | VV_VH_ratio  | Derived     | dB      | VV − VH (surface roughness proxy) |
| 5     | HAND         | MERIT Hydro | metres  | Height above nearest drainage     |

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
11. **6-band model input** — any reference to "7-band" in older comments is outdated. The correct number is **6**.

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
| uncertainty.py    | 4     | ✓ Complete |
| 06_exposure.py    | 5     | ✓ Complete |
| eval.py           | 6     | ✓ Complete |
| plots.py          | 6     | ✓ Complete |
| run_experiment.py | —     | ✓ Complete |
