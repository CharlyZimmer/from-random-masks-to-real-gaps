#!/usr/bin/env bash
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate.sh for tracked runs).
set -euo pipefail
PYTHONPATH=. python sentinel/src/evaluation/evaluate_linear_interpolation_baseline.py \
    --data_dir   "${S2_DATA_DIR:-sentinel/data}/gap_filling" \
    --n_jobs     2 \
    --normalize \
    --num_samples 3
