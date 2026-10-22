#!/usr/bin/env bash
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate.sh for tracked runs).
set -euo pipefail
PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${ESDC_GAP_FILLING_DIR:-esdc/data/lst_gap_filling}/dataset.h5" \
    --log_dir    "esdc/lightning_logs/experiments/exp04_scratch_gapfilling" \
    --device     cpu \
    --batch_size 4 \
    --num_samples 20 "$@"
