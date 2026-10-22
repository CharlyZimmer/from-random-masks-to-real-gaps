#!/usr/bin/env bash
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate.sh for tracked runs).
set -euo pipefail
PYTHONPATH=. python esdc/src/evaluation/evaluate_kriging_baseline.py \
    --data_dir   "${ESDC_GAP_FILLING_DIR:-esdc/data/lst_gap_filling}/dataset.h5" \
    --variogram_model linear \
    --n_jobs     4 \
    --num_samples 20
