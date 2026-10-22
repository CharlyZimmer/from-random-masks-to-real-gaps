#!/usr/bin/env bash
set -euo pipefail

LMDB_PATH="${S2_LMDB_PATH:-./sentinel/data/ssl4eo_consecutive_s2c.lmdb}"

PYTHONPATH=. python sentinel/src/training/gap_aware_training.py \
  --lmdb_path "$LMDB_PATH" \
  --patch_size 8 \
  --batch_size 2 \
  --num_workers 2 \
  --max_epochs 2 \
  --accelerator cpu \
  --devices 1 \
  --precision 32
