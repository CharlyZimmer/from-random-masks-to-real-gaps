#!/usr/bin/env bash
# Structured-masking zero-shot evaluation — run from project root.
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate.sh for tracked runs).
set -euo pipefail

PYTHONPATH=. python sentinel/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${S2_DATA_DIR:-sentinel/data}/gap_filling" \
    --log_dir    "sentinel/lightning_logs/experiments/exp03_structured_masking" \
    --device     cpu \
    --batch_size 2 \
    --normalize \
    --num_samples 10 "$@"
