#!/usr/bin/env bash
# exp03: Pre-training with BEiT-style structured masking (for subsequent zero-shot eval).
# Sanity-check run: cpu, tiny model, 2 epochs.
# Full HPC run: use scripts/slurm/train.sh instead.
set -euo pipefail

PYTHONPATH=. python sentinel/experiments/scripts/run_training.py \
    --config      sentinel/experiments/exp03_structured_masking/config.yaml \
    --accelerator cpu \
    --devices     1 \
    --precision   32 \
    --embed_dim   192 \
    --depth       1 \
    --num_heads   1 \
    --patch_size  8 \
    --max_epochs  2 \
    --max_samples 10
