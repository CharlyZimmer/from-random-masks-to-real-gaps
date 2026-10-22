#!/usr/bin/env bash
#SBATCH --job-name=interp_baseline_eval
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=esdc/slurm_logs/interp_baseline_eval_%j.out
set -euo pipefail
EXP_DIR="${GAP_AWARE_ROOT:-.}/esdc/experiments/interpolation_baseline"
PYTHONPATH=. python esdc/src/evaluation/evaluate_kriging_baseline.py \
    --data_dir      "${ESDC_GAP_FILLING_DIR}/dataset.h5" \
    --variogram_model linear \
    --n_jobs        16 \
    --results_dir   "${EXP_DIR}/results" \
    --experiment_id "interpolation_baseline"
