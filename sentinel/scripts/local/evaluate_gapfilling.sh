#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${S2_DATA_DIR:-./sentinel/data}/gap_filling"

PYTHONPATH=. python sentinel/src/evaluation/evaluate_gapfilling.py \
  --data_dir "$DATA_DIR" \
  --log_dir  ./sentinel/lightning_logs/pretraining \
  --device   cpu \
  --normalize \
  --num_samples 10
