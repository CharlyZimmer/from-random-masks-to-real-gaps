# ESDC Gap-Filling Experiments

This directory contains one subdirectory per experiment. All experiments compare methods on the same canonical test set (`$ESDC_GAP_FILLING_DIR/dataset.h5` — see `eval_dataset.yaml`).

## Experiments

| ID | Method | Type |
|----|--------|------|
| `exp01_zero_shot` | Pretrained MaskFeat evaluated directly | zero-shot inference |
| `exp02_finetuning` | Pretrained → fine-tuned on gap-filling data | fine-tuning |
| `exp03_structured_masking` | Same as exp01, BEiT-style block masking instead of random | masking ablation of exp01 |
| `exp04_scratch_gapfilling` | Train from scratch on gap-filling data | baseline |
| `interpolation_baseline` | Ordinary Kriging spatial interpolation (no learned model) | non-learned baseline |
| `linear_interpolation_baseline` | Linear spatial interpolation (no learned model) | non-learned baseline |

Some experiments have additional nested ablations under their own directory
(not listed in the table above, each with its own `README.md`/`results/RESULTS.md`):
- `exp01_zero_shot/max_gap_ablation/` — pre-training-dataset max-gap-ratio sweep (0%, 10%, vs. the 20% baseline), via post-hoc LMDB filtering (no dataset regeneration).
- `exp01_zero_shot/model_size_ablation/` — model-size sweep (6 progressively smaller ViT configs vs. the ViT-Ti 192/12/3 baseline, which is reused rather than retrained).
- `exp02_finetuning/sample_ablation/` — fine-tuning sample-count sweep.

## Per-experiment layout

```
expXX_<name>/
├── config.yaml               # authoritative: all dataset, model, training parameters
├── scripts/
│   ├── local/
│   │   ├── train.sh          # sanity check (cpu, tiny model, 100 samples, 2 epochs)
│   │   └── evaluate.sh       # local eval (cpu, 20 samples); prints metrics only, does not write results/
│   └── slurm/
│       ├── train.sh          # HPC job using config.yaml defaults
│       └── evaluate.sh       # HPC eval (full dataset); writes results/
└── results/
    ├── metrics.json          # written by scripts/slurm/evaluate.sh; committed to git
    └── run_info.json         # checkpoint path, git commit, eval date; committed to git
```

Nested ablations that post-hoc filter an existing dataset (e.g. `exp01_zero_shot/max_gap_ablation/`) reuse the parent experiment's dataset unchanged and filter it at training time.

`interpolation_baseline/` (Ordinary Kriging) and `linear_interpolation_baseline/` (linear interpolation) are non-learned baselines and have no `train.sh` — there's no fitted/persisted state, only `scripts/{local,slurm}/evaluate.sh`.

## Workflow

### Adding a new experiment
1. Copy the nearest existing experiment directory; rename to `expXX_<slug>`.
2. Edit `config.yaml`: set `experiment.id`, `description`, `type`, and any changed parameters.
3. Sanity-check locally: `./esdc/experiments/expXX_<slug>/scripts/local/train.sh`
4. Submit HPC job: `sbatch esdc/experiments/expXX_<slug>/scripts/slurm/train.sh`
5. After training: `sbatch esdc/experiments/expXX_<slug>/scripts/slurm/evaluate.sh`
6. Commit `results/metrics.json` and `results/run_info.json`.
7. Regenerate the comparison table and commit:
   ```
   python esdc/experiments/scripts/collect_results.py
   git add esdc/experiments/
   git commit -m "exp XX results: ..."
   ```

### Running from the project root
All scripts are designed to be run from the repository root (`gap-aware/`):
```bash
# Local sanity check:
./esdc/experiments/exp04_scratch_gapfilling/scripts/local/train.sh

# HPC submission:
sbatch esdc/experiments/exp04_scratch_gapfilling/scripts/slurm/train.sh
```

## Config → CLI bridge

`esdc/experiments/scripts/run_training.py` reads `config.yaml` and assembles the `gap_aware_training.py` CLI call. Local train scripts pass override flags for reduced-scale runs:

```bash
python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp04_scratch_gapfilling/config.yaml \
    --accelerator cpu --devices 1 --precision 32 \
    --embed_dim 192 --depth 1 --num_heads 1 --patch_size 8 \
    --max_epochs 2 --max_samples 100
```

HPC Slurm scripts omit all override flags — the config defaults are used.

## Results

See [RESULTS.md](RESULTS.md) for the current comparison table.
Regenerate with:
```
python esdc/experiments/scripts/collect_results.py
```
The script writes `RESULTS.md` directly.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ESDC_GAP_FILLING_DIR` | `esdc/data/lst_gap_filling` | Gap-filling evaluation set (`dataset.h5`) |
| `ESDC_GAP_FILLING_TRAINING_DIR` | `esdc/data/lst_gap_filling_training` | Gap-filling training set (`dataset.h5`, `lmdb/`) |

`exp02_finetuning` always fine-tunes from `exp01_zero_shot`'s own checkpoint, auto-discovered from `esdc/lightning_logs/experiments/exp01_zero_shot` (lowest `MAE_val`) — it does not use `ESDC_PRETRAINED_CKPT`. That variable belongs to the separate, non-experiment-tracked pretraining pipeline (`esdc/scripts/*/training.sh`, a different 384/8/6 architecture); see `.env.example` for details.

See `.env.example` at the project root for all variables.
