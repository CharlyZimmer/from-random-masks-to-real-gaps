#!/usr/bin/env bash
#SBATCH --job-name=exp01_model_size_ablation_eval_all
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=esdc/slurm_logs/exp01_model_size_ablation_eval_all_%j.out
# Zero-shot evaluation of exp01_zero_shot's model-size ablation — all 6
# smaller variants, run as sequential job steps inside a SINGLE 1-GPU
# allocation (evaluation is always single-GPU regardless of training
# devices, so all 6 fold into one loop cleanly; see evaluate_<label>.sh for
# the equivalent one-variant-per-job scripts).
#
# The (192,12,3) baseline is NOT re-evaluated here — it reuses
# exp01_zero_shot/results/ directly (see ../../README.md).
#
# Submit from project root, after train_all.sh has finished:
# sbatch esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/slurm/evaluate_all.sh
set -euo pipefail

EXP_DIR="${GAP_AWARE_ROOT:-.}/esdc/experiments/exp01_zero_shot/model_size_ablation"
LOG_ROOT="esdc/lightning_logs/experiments/exp01_zero_shot/model_size_ablation"

for label in 192_12_2 192_6_2 96_6_2 96_6_1 96_3_1 48_3_1; do
    echo "=== model_size_ablation evaluate_all: ${label} ==="
    PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
        --data_dir   "${ESDC_GAP_FILLING_DIR}/dataset.h5" \
        --log_dir    "${LOG_ROOT}/${label}" \
        --device     cuda \
        --batch_size 16 \
        --results_dir  "${EXP_DIR}/results/${label}" \
        --experiment_id "exp01_zero_shot_${label}" \
        --full_only "$@"
done
