#!/usr/bin/env bash
#SBATCH --job-name=exp02_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=sentinel/slurm_logs/exp02_eval_%j.out
# Fine-tuning evaluation — submit from project root: sbatch sentinel/experiments/exp02_finetuning/scripts/slurm/evaluate.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

EXP_DIR="${GAP_AWARE_ROOT:-.}/sentinel/experiments/exp02_finetuning"

PYTHONPATH=. srun python sentinel/src/evaluation/evaluate_gapfilling.py \
    --data_dir   "${S2_DATA_DIR}/gap_filling" \
    --log_dir    "sentinel/lightning_logs/experiments/exp02_finetuning" \
    --device     cuda \
    --batch_size 8 \
    --normalize \
    --results_dir  "${EXP_DIR}/results" \
    --experiment_id "exp02_finetuning" "$@"
