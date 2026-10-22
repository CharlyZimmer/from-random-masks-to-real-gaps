#!/usr/bin/env bash
set -euo pipefail

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
  --log_dir esdc/lightning_logs/pretraining
