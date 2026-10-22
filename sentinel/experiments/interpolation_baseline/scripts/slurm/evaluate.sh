#!/usr/bin/env bash
#SBATCH --job-name=interp_baseline_eval
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sentinel/slurm_logs/interp_baseline_eval_%j.out
# Per-band Kriging over 128x128x10 bands is much heavier than ESDC's 64x64x1
# baseline — 1 day walltime vs. ESDC's 4 hours; tune --max_obs_points down if
# this doesn't fit.
set -euo pipefail
EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/interpolation_baseline"
PYTHONPATH=. python sentinel/src/evaluation/evaluate_kriging_baseline.py \
    --data_dir      "${S2_DATA_DIR}/gap_filling" \
    --variogram_model linear \
    --max_obs_points 2000 \
    --n_jobs        16 \
    --normalize \
    --results_dir   "${EXP_DIR}/results" \
    --experiment_id "interpolation_baseline"
