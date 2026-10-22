#!/usr/bin/env bash
#SBATCH --job-name=exp04_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=esdc/slurm_logs/exp04_eval_%j.out
set -euo pipefail
EXP_DIR="${GAP_AWARE_ROOT:-.}/esdc/experiments/exp04_scratch_gapfilling"
PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${ESDC_GAP_FILLING_DIR}/dataset.h5" \
    --log_dir    "esdc/lightning_logs/experiments/exp04_scratch_gapfilling" \
    --device     cuda \
    --batch_size 16 \
    --results_dir  "${EXP_DIR}/results" \
    --experiment_id "exp04_scratch_gapfilling" "$@"
