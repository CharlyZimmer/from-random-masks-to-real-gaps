#!/usr/bin/env bash
#SBATCH --job-name=exp02_n6400_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:30:00
#SBATCH --output=sentinel/slurm_logs/exp02_n6400_eval_%j.out
# Sample-efficiency ablation evaluation -- n6400 (6400 finetuning samples),
# evaluated on the same canonical eval set ($S2_DATA_DIR/gap_filling) as
# every other experiment.
# Submit from project root: sbatch sentinel/experiments/exp02_finetuning/sample_ablation/scripts/slurm/evaluate_n6400.sh
set -euo pipefail

EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/exp02_finetuning/sample_ablation"

PYTHONPATH=. python sentinel/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${S2_DATA_DIR}/gap_filling" \
    --log_dir    "sentinel/lightning_logs/experiments/exp02_finetuning/sample_ablation/n6400" \
    --device     cuda \
    --batch_size 8 \
    --normalize \
    --results_dir  "${EXP_DIR}/results/n6400" \
    --experiment_id "exp02_finetuning_n6400" "$@"
