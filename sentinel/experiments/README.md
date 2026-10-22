# Sentinel-2 Gap-Filling Experiments

This directory contains one subdirectory per experiment. All experiments compare methods on the same canonical test set (`$S2_DATA_DIR/gap_filling`). Structurally a direct port of `esdc/experiments/`, adapted for Sentinel's differences: 128x128 images and all 10 bands reconstructed (no single "primary variable + aux" concept). Both modules use the same ViT-Ti model (embed_dim 192, depth 12, num_heads 3).

## Experiments

| ID | Method | Type |
|----|--------|------|
| `exp01_zero_shot` | Pretrained MaskFeat evaluated directly | zero-shot inference |
| `exp02_finetuning` | Pretrained → fine-tuned on gap-filling training data | fine-tuning |
| `exp03_structured_masking` | Same as exp01, BEiT-style block masking instead of random | masking ablation of exp01 |
| `exp04_scratch_gapfilling` | Train from scratch on gap-filling training data | baseline |
| `interpolation_baseline` | Ordinary Kriging spatial interpolation, per band (no learned model) | non-learned baseline |
| `linear_interpolation_baseline` | Linear spatial interpolation, per band (no learned model) | non-learned baseline |

## Per-experiment layout

```
expXX_<name>/
├── config.yaml               # authoritative: all dataset, model, training parameters
├── scripts/
│   ├── local/
│   │   ├── train.sh          # sanity check (cpu, tiny model, ~10 samples, 2 epochs)
│   │   └── evaluate.sh       # local eval (cpu, ~10 samples); prints metrics only, does not write results/
│   └── slurm/
│       ├── train.sh          # HPC job using config.yaml defaults
│       └── evaluate.sh       # HPC eval (full dataset); writes results/
└── results/
    ├── metrics.json          # written by scripts/slurm/evaluate.sh; committed to git
    └── run_info.json         # checkpoint path, git commit, eval date; committed to git
```

`interpolation_baseline/` (Ordinary Kriging) and `linear_interpolation_baseline/` (linear interpolation) are non-learned baselines and have no `train.sh` — there's no fitted/persisted state, only `scripts/{local,slurm}/evaluate.sh`.

Some experiments have additional nested ablations under their own directory
(same per-experiment layout as above, plus a `configs/` subdirectory holding
one config per variant):

- `exp01_zero_shot/max_gap_ablation/` — pre-training-LMDB max-gap-ratio sweep (0%, 10%, vs. the 20% baseline), via post-hoc LMDB filtering at training time (no dataset regeneration).
- `exp02_finetuning/sample_ablation/` — fine-tuning sample-count sweep.

## Workflow

### Adding a new experiment
1. Copy the nearest existing experiment directory; rename to `expXX_<slug>`.
2. Edit `config.yaml`: set `experiment.id`, `description`, `type`, and any changed parameters.
3. Sanity-check locally: `./sentinel/experiments/expXX_<slug>/scripts/local/train.sh`
4. Submit HPC job: `sbatch sentinel/experiments/expXX_<slug>/scripts/slurm/train.sh`
5. After training: run `./sentinel/experiments/expXX_<slug>/scripts/slurm/evaluate.sh`
6. Commit `results/metrics.json` and `results/run_info.json`.
7. Regenerate the comparison table and commit:
   ```
   python sentinel/experiments/scripts/collect_results.py
   git add sentinel/experiments/
   git commit -m "exp XX results: ..."
   ```

### Running from the project root
All scripts are designed to be run from the repository root (`gap-aware/`):
```bash
# Local sanity check:
./sentinel/experiments/exp04_scratch_gapfilling/scripts/local/train.sh

# HPC submission:
sbatch sentinel/experiments/exp04_scratch_gapfilling/scripts/slurm/train.sh
```

## Config → CLI bridge

`sentinel/experiments/scripts/run_training.py` reads `config.yaml` and assembles the `gap_aware_training.py` CLI call. Local train scripts pass override flags for reduced-scale runs:

```bash
python sentinel/experiments/scripts/run_training.py \
    --config sentinel/experiments/exp04_scratch_gapfilling/config.yaml \
    --accelerator cpu --devices 1 --precision 32 \
    --embed_dim 192 --depth 1 --num_heads 1 --patch_size 8 \
    --max_epochs 2 --max_samples 10
```

HPC Slurm scripts omit all override flags — the config defaults are used.

## Results

See [RESULTS.md](RESULTS.md) for the current comparison table.
Regenerate with:
```
python sentinel/experiments/scripts/collect_results.py
```
The script writes `RESULTS.md` directly; redirecting stdout is not required.

## Environment variables

| Variable | Local default | Purpose |
|---|---|---|
| `S2_DATA_DIR` | `sentinel/data` | Contains `SSL-consecutive/` (pretraining) and `gap_filling/` (canonical eval set, all four learned experiments + both baselines evaluate against `$S2_DATA_DIR/gap_filling`) |
| `S2_LMDB_PATH` | `sentinel/data/ssl4eo_consecutive_s2c.lmdb` | Pretraining LMDB used by `exp01_zero_shot`/`exp03_structured_masking` |
| `S2_GAP_FILLING_TRAINING_DIR` | `sentinel/data/gap_filling_training` | Separate download used only for `exp02_finetuning`/`exp04_scratch_gapfilling` training — reuses the same SSL4EO-S12 locations as pretraining (`--match_file`), with synthetic gaps injected; never used for evaluation, so the RESULTS.md table stays comparable across all experiments (mirrors ESDC's `ESDC_GAP_FILLING_TRAINING_DIR`) |

`exp02_finetuning` always fine-tunes from `exp01_zero_shot`'s own checkpoint, auto-discovered from `sentinel/lightning_logs/experiments/exp01_zero_shot` (lowest `MAE_val`) — it does not use the separate, non-experiment-tracked standalone pretraining pipeline (`sentinel/scripts/*/training.sh`, a different 384/8/6 architecture).

Unlike ESDC, Sentinel has no single physical unit across its 10 bands (no Kelvin-equivalent), so `metrics.json`/`RESULTS.md` report only normalized ([0,1] reflectance) MAE.

See `.env.example` at the project root for all variables.
