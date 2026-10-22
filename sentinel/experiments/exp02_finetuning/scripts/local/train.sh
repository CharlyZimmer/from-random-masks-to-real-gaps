#!/usr/bin/env bash
# Fine-tuning sanity check (cpu, tiny model, small dataset) — run from project root.
# batch_size kept small: a small local S2_GAP_FILLING_TRAINING_DIR may hold
# only a handful of locations, so a larger batch would drop every batch
# (drop_last=True) and train on zero steps.
set -euo pipefail

PYTHONPATH=. python sentinel/experiments/scripts/run_training.py \
    --config      sentinel/experiments/exp02_finetuning/config.yaml \
    --accelerator cpu \
    --devices     1 \
    --precision   32 \
    --embed_dim   192 \
    --depth       1 \
    --num_heads   1 \
    --patch_size  8 \
    --max_epochs  2 \
    --batch_size  2 \
    --max_samples 10
