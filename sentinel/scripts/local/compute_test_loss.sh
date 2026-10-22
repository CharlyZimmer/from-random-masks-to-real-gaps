#!/usr/bin/env bash
set -euo pipefail

LMDB_PATH="${S2_LMDB_PATH:-./sentinel/data/ssl4eo_consecutive_s2c.lmdb}"
TEST_LMDB="${LMDB_PATH%.lmdb}_test.lmdb"

PYTHONPATH=. python sentinel/src/training/compute_test_loss.py \
  --test_lmdb_path "$TEST_LMDB" \
  --batch_size 2 \
  --num_workers 2 \
  --prefetch_factor 2 \
  --accelerator cpu \
  --devices 1 \
  --precision 32
