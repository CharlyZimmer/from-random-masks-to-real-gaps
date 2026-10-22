#!/usr/bin/env bash
#SBATCH --job-name=exp02_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=esdc/slurm_logs/exp02_eval_%j.out
# Fine-tuning evaluation — submit from project root: sbatch esdc/experiments/exp02_finetuning/scripts/slurm/evaluate.sh
set -euo pipefail

EXP_DIR="${GAP_AWARE_ROOT:-.}/esdc/experiments/exp02_finetuning"

PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${ESDC_GAP_FILLING_DIR}/dataset.h5" \
    --log_dir    "esdc/lightning_logs/experiments/exp02_finetuning" \
    --device     cuda \
    --batch_size 16 \
    --results_dir  "${EXP_DIR}/results" \
    --experiment_id "exp02_finetuning" "$@"
