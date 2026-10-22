#!/usr/bin/env bash
# Training-from-scratch sanity check (cpu, tiny model) — run from project root.
set -euo pipefail

PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
    --config      esdc/experiments/exp04_scratch_gapfilling/config.yaml \
    --accelerator cpu \
    --devices     1 \
    --precision   32 \
    --embed_dim   192 \
    --depth       1 \
    --num_heads   1 \
    --patch_size  8 \
    --max_epochs  2 \
    --max_samples 100
