#!/usr/bin/env bash
#SBATCH --job-name=linear_interp_baseline_eval
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=sentinel/slurm_logs/linear_interp_baseline_eval_%j.out
set -euo pipefail
EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/linear_interpolation_baseline"
PYTHONPATH=. python sentinel/src/evaluation/evaluate_linear_interpolation_baseline.py \
    --data_dir      "${S2_DATA_DIR}/gap_filling" \
    --n_jobs        16 \
    --normalize \
    --results_dir   "${EXP_DIR}/results" \
    --experiment_id "linear_interpolation_baseline"
