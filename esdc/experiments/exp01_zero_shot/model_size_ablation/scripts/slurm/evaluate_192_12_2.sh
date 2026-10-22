#!/usr/bin/env bash
#SBATCH --job-name=exp01_192_12_2_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=esdc/slurm_logs/exp01_192_12_2_eval_%j.out
# model_size_ablation (192/12/2) zero-shot evaluation — submit from project root:
# sbatch esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/slurm/evaluate_192_12_2.sh
set -euo pipefail

EXP_DIR="${GAP_AWARE_ROOT:-.}/esdc/experiments/exp01_zero_shot/model_size_ablation"

PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${ESDC_GAP_FILLING_DIR}/dataset.h5" \
    --log_dir    "esdc/lightning_logs/experiments/exp01_zero_shot/model_size_ablation/192_12_2" \
    --device     cuda \
    --batch_size 16 \
    --results_dir  "${EXP_DIR}/results/192_12_2" \
    --experiment_id "exp01_zero_shot_192_12_2" \
    --full_only "$@"
