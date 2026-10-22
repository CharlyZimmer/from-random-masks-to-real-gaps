#!/usr/bin/env bash
set -euo pipefail

DATA_H5="${ESDC_GAP_FILLING_DIR:-./esdc/data/lst_gap_filling}/dataset.h5"

PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
  --data_dir  "$DATA_H5" \
  --log_dir   ./esdc/lightning_logs/pretraining \
  --device    cpu \
  --num_samples 10
