#!/usr/bin/env bash
# max_gap_ablation: Pre-training sanity check for both gap-ratio variants (0%, 10%).
# Sanity-check run: cpu, tiny model, 2 epochs, 10 samples (post-filter).
# Full HPC run: use scripts/slurm/train_gap000.sh / train_gap010.sh instead.
set -euo pipefail

CFG_DIR="sentinel/experiments/exp01_zero_shot/max_gap_ablation/configs"

for label in gap000 gap010; do
    echo "=== max_gap_ablation local sanity: ${label} ==="
    PYTHONPATH=. python sentinel/experiments/scripts/run_training.py \
        --config      "${CFG_DIR}/${label}.yaml" \
        --accelerator cpu \
        --devices     1 \
        --precision   32 \
        --embed_dim   192 \
        --depth       1 \
        --num_heads   1 \
        --patch_size  8 \
        --max_epochs  2 \
        --max_samples 10
done
