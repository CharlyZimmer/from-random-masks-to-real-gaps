#!/usr/bin/env bash
#SBATCH --job-name=exp02_n0100_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=esdc/slurm_logs/exp02_n0100_eval_%j.out
# Sample-efficiency ablation evaluation -- n0100 (100 finetuning samples),
# evaluated on the same canonical eval set as every other experiment.
# Submit from project root: sbatch esdc/experiments/exp02_finetuning/sample_ablation/scripts/slurm/evaluate_n0100.sh
set -euo pipefail

EXP_DIR="${GAP_AWARE_ROOT:-.}/esdc/experiments/exp02_finetuning/sample_ablation"

PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${ESDC_GAP_FILLING_DIR}/dataset.h5" \
    --log_dir    "esdc/lightning_logs/experiments/exp02_finetuning/sample_ablation/n0100" \
    --device     cuda \
    --batch_size 16 \
    --results_dir  "${EXP_DIR}/results/n0100" \
    --experiment_id "exp02_finetuning_n0100" "$@"
