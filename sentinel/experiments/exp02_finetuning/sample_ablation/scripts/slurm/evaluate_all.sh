#!/usr/bin/env bash
#SBATCH --job-name=exp02_ft_ablation_eval_all
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-16:00:00
#SBATCH --output=sentinel/slurm_logs/exp02_ft_ablation_eval_all_%j.out
# Sample-efficiency ablation evaluation -- all 9 checkpoints (n0050/n0100/
# n0200/n0400/n0800/n1600/n3200/n6400/n12800) evaluated as sequential steps
# inside a SINGLE resource allocation, so the 1-GPU reservation is only
# queued for once instead of 9 separate times (see evaluate_n*.sh for the
# equivalent one-variant-per-job scripts, useful for re-evaluating a single
# variant in isolation). Evaluation is always single-GPU regardless of how
# many devices a variant was trained with, so all 9 fold into the same loop
# cleanly. Each variant is evaluated on the same canonical eval set
# ($S2_DATA_DIR/gap_filling) as every other experiment. Run after
# train_all.sh + train_small_all.sh (or the individual train_n*.sh jobs) have
# produced all 9 checkpoints.
#
# --time is the sum of the individual evaluate_n*.sh budgets (9x1.5h=13.5h)
# plus a buffer for 9x job-step startup overhead.
#
# Submit from project root: sbatch sentinel/experiments/exp02_finetuning/sample_ablation/scripts/slurm/evaluate_all.sh
set -euo pipefail

EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/exp02_finetuning/sample_ablation"
LOG_ROOT="sentinel/lightning_logs/experiments/exp02_finetuning/sample_ablation"

for label in n0050 n0100 n0200 n0400 n0800 n1600 n3200 n6400 n12800; do
    echo "=== sample_ablation evaluate_all: ${label} ==="
    PYTHONPATH=. python sentinel/src/evaluation/evaluate_gapfilling.py \
        --data_dir   "${S2_DATA_DIR}/gap_filling" \
        --log_dir    "${LOG_ROOT}/${label}" \
        --device     cuda \
        --batch_size 8 \
        --normalize \
        --results_dir  "${EXP_DIR}/results/${label}" \
        --experiment_id "exp02_finetuning_${label}" "$@"
done
