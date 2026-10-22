#!/usr/bin/env bash
# model_size_ablation: Zero-shot evaluation sanity check for all 6 smaller-model variants — run from project root.
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate_all.sh for tracked runs).
set -euo pipefail

LOG_ROOT="esdc/lightning_logs/experiments/exp01_zero_shot/model_size_ablation"

for label in 192_12_2 192_6_2 96_6_2 96_6_1 96_3_1 48_3_1; do
    echo "=== model_size_ablation local eval sanity: ${label} ==="
    PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
        --data_dir   "${ESDC_GAP_FILLING_DIR:-esdc/data/lst_gap_filling}/dataset.h5" \
        --log_dir    "${LOG_ROOT}/${label}" \
        --device     cpu \
        --batch_size 4 \
        --num_samples 20 \
        --full_only "$@"
done
