#!/usr/bin/env bash
#SBATCH --job-name=exp01_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=sentinel/slurm_logs/exp01_eval_%j.out
# Zero-shot evaluation — submit from project root: sbatch sentinel/experiments/exp01_zero_shot/scripts/slurm/evaluate.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/exp01_zero_shot"

PYTHONPATH=. srun python sentinel/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${S2_DATA_DIR}/gap_filling" \
    --log_dir    "sentinel/lightning_logs/experiments/exp01_zero_shot" \
    --device     cuda \
    --batch_size 8 \
    --normalize \
    --results_dir  "${EXP_DIR}/results" \
    --experiment_id "exp01_zero_shot" \
    --checkpoint "sentinel/lightning_logs/experiments/exp01_zero_shot/version_3/checkpoints/gapfiller-epoch=45-MAE_val=0.0097.ckpt" "$@"
