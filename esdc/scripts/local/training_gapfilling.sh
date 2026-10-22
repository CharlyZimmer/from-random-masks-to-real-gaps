#!/usr/bin/env bash
set -euo pipefail

export ESDC_DATA_DIR="${ESDC_GAP_FILLING_TRAINING_DIR:-./esdc/data/lst_gap_filling_training}"

PYTHONPATH=. python esdc/src/training/gap_aware_training.py \
  --patch_size 8 \
  --batch_size 8 \
  --num_workers 2 \
  --max_epochs 2 \
  --max_samples 100 \
  --warmup_steps 50 \
  --accelerator cpu \
  --devices 1 \
  --precision 32 \
  --gap_only_masking \
  --y_lmdb_path "${ESDC_DATA_DIR}/lmdb/dataset_y.lmdb" \
  --log_dir esdc/lightning_logs/gapfilling_training
