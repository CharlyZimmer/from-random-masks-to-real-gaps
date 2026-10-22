#!/usr/bin/env bash
# Sample-efficiency ablation sanity check (cpu, tiny model, 10 samples, 2 epochs
# per variant) — run from project root.
#
# Loops all 9 sample-count configs. --devices 1 is forced on the CLI for every
# variant (overriding each config's own devices=1/4), since this is a
# single-process CPU sanity check, not a DDP run. This only verifies that each
# of the 9 differently-configured variants (different log_dir, warmup_steps,
# devices, base_lr) trains without error -- it does NOT approximate the real
# target N (50-12800), which a small local gap_masks.lmdb cannot reach. Real
# numbers only come from the scripts/slurm/ jobs.
set -euo pipefail

CFG_DIR="sentinel/experiments/exp02_finetuning/sample_ablation/configs"

for label in n0050 n0100 n0200 n0400 n0800 n1600 n3200 n6400 n12800; do
    echo "=== sample_ablation local sanity: ${label} ==="
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
        --batch_size  2 \
        --max_samples 10
done
