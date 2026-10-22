#!/usr/bin/env bash
# Sample-efficiency ablation sanity check (cpu, tiny model, 100 samples, 2 epochs
# per variant) — run from project root.
#
# Loops all 9 sample-count configs. This only verifies that each of the 9
# differently-configured variants (different log_dir, warmup_steps, and for
# n0050/n0100/n0200 also devices/base_lr) trains without error — it does NOT
# approximate the real target N (50-12800): the local gap-filling-training
# LMDB has only 806 entries total, so N>=800 is not reachable locally. Real
# numbers only come from the scripts/slurm/ jobs on the HPC.
set -euo pipefail

CFG_DIR="esdc/experiments/exp02_finetuning/sample_ablation/configs"

for label in n0050 n0100 n0200 n0400 n0800 n1600 n3200 n6400 n12800; do
    echo "=== sample_ablation local sanity: ${label} ==="
    PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
        --config      "${CFG_DIR}/${label}.yaml" \
        --accelerator cpu \
        --devices     1 \
        --precision   32 \
        --embed_dim   192 \
        --depth       1 \
        --num_heads   1 \
        --patch_size  8 \
        --max_epochs  2 \
        --max_samples 100
done
