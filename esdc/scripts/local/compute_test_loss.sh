#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python esdc/src/training/compute_test_loss.py \
  --batch_size 8 \
  --num_workers 2 \
  --accelerator cpu \
  --devices 1 \
  --precision 32
