#!/usr/bin/env bash
#SBATCH --job-name=exp01_gap000_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=sentinel/slurm_logs/exp01_gap000_eval_%j.out
# max_gap_ablation (0%) zero-shot evaluation — submit from project root:
# sbatch sentinel/experiments/exp01_zero_shot/max_gap_ablation/scripts/slurm/evaluate_gap000.sh
set -euo pipefail

EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/exp01_zero_shot/max_gap_ablation"

PYTHONPATH=. python sentinel/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${S2_DATA_DIR}/gap_filling" \
    --log_dir    "sentinel/lightning_logs/experiments/exp01_zero_shot/max_gap_ablation/gap000" \
    --device     cuda \
    --batch_size 8 \
    --normalize \
    --results_dir  "${EXP_DIR}/results/gap000" \
    --experiment_id "exp01_zero_shot_gap000" "$@"
