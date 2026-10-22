#!/usr/bin/env bash
# exp01: Pre-training at experiment model size (for subsequent zero-shot eval).
# Sanity-check run: cpu, tiny model, 2 epochs.
# Full HPC run: use scripts/slurm/train.sh instead.
set -euo pipefail

PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
    --config      esdc/experiments/exp01_zero_shot/config.yaml \
    --accelerator cpu \
    --devices     1 \
    --precision   32 \
    --embed_dim   192 \
    --depth       1 \
    --num_heads   1 \
    --patch_size  8 \
    --max_epochs  2 \
    --max_samples 100
