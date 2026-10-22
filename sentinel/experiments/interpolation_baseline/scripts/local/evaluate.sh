#!/usr/bin/env bash
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate.sh for tracked runs).
# Small max_obs_points/num_samples — Kriging at 128x128x10 bands is much more
# expensive than ESDC's 64x64x1 (see evaluate_kriging_baseline.py).
set -euo pipefail
PYTHONPATH=. python sentinel/src/evaluation/evaluate_kriging_baseline.py \
    --data_dir   "${S2_DATA_DIR:-sentinel/data}/gap_filling" \
    --variogram_model linear \
    --max_obs_points 500 \
    --n_jobs     2 \
    --normalize \
    --num_samples 3
