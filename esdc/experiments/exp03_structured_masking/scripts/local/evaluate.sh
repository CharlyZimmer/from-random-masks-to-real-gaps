#!/usr/bin/env bash
# Zero-shot evaluation — run from project root.
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate.sh for tracked runs).
set -euo pipefail

PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${ESDC_GAP_FILLING_DIR:-esdc/data/lst_gap_filling}/dataset.h5" \
    --log_dir    "esdc/lightning_logs/experiments/exp03_structured_masking" \
    --device     cpu \
    --batch_size 4 \
    --num_samples 20 "$@"
