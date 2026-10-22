#!/usr/bin/env bash
# model_size_ablation: Pre-training sanity check for all 6 smaller-model variants.
# Sanity-check run: cpu, 2 epochs, 100 samples. Unlike other ablations' local
# scripts, this one does NOT override --embed_dim/--depth/--num_heads --
# architecture is exactly what's being ablated here, so each variant's config
# values must be used as-is to prove that specific architecture builds and
# trains without error.
# Full HPC run: use scripts/slurm/train_all.sh instead.
set -euo pipefail

CFG_DIR="esdc/experiments/exp01_zero_shot/model_size_ablation/configs"

for label in 192_12_2 192_6_2 96_6_2 96_6_1 96_3_1 48_3_1; do
    echo "=== model_size_ablation local sanity: ${label} ==="
    PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
        --config      "${CFG_DIR}/${label}.yaml" \
        --accelerator cpu \
        --devices     1 \
        --precision   32 \
        --max_epochs  2 \
        --max_samples 100
done
