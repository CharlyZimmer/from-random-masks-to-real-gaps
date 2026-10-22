#!/usr/bin/env bash
# Sample-efficiency ablation evaluation sanity check (cpu) — run from project root.
# Local sanity check only: prints metrics, does not write results/ (use
# scripts/slurm/evaluate_n*.sh for tracked runs).
set -euo pipefail

LOG_ROOT="esdc/lightning_logs/experiments/exp02_finetuning/sample_ablation"

for label in n0050 n0100 n0200 n0400 n0800 n1600 n3200 n6400 n12800; do
    echo "=== sample_ablation local eval sanity: ${label} ==="
    PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
        --data_dir   "${ESDC_GAP_FILLING_DIR:-esdc/data/lst_gap_filling}/dataset.h5" \
        --log_dir    "${LOG_ROOT}/${label}" \
        --device     cpu \
        --batch_size 4 \
        --num_samples 20 "$@"
done
