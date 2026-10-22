#!/usr/bin/env bash
#SBATCH --job-name=exp04_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=sentinel/slurm_logs/exp04_eval_%j.out
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/exp04_scratch_gapfilling"
PYTHONPATH=. srun python sentinel/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${S2_DATA_DIR}/gap_filling" \
    --log_dir    "sentinel/lightning_logs/experiments/exp04_scratch_gapfilling" \
    --device     cuda \
    --batch_size 8 \
    --normalize \
    --results_dir  "${EXP_DIR}/results" \
    --experiment_id "exp04_scratch_gapfilling" \
    --checkpoint "sentinel/lightning_logs/experiments/exp04_scratch_gapfilling/version_3/checkpoints/gapfiller-epoch=48-MAE_val=0.0147.ckpt" "$@"
