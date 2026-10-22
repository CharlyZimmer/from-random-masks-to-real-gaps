#!/usr/bin/env bash
# max_gap_ablation: Zero-shot evaluation sanity check for both gap-ratio variants — run from project root.
# Local sanity check only: prints metrics, does not write results/ (use scripts/slurm/evaluate_gap*.sh for tracked runs).
set -euo pipefail

LOG_ROOT="sentinel/lightning_logs/experiments/exp01_zero_shot/max_gap_ablation"

for label in gap000 gap010; do
    echo "=== max_gap_ablation local eval sanity: ${label} ==="
    PYTHONPATH=. python sentinel/src/evaluation/evaluate_gapfilling.py \
        --data_dir   "${S2_DATA_DIR:-sentinel/data}/gap_filling" \
        --log_dir    "${LOG_ROOT}/${label}" \
        --device     cpu \
        --batch_size 2 \
        --normalize \
        --num_samples 10 "$@"
done
