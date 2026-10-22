#!/usr/bin/env bash
#SBATCH --job-name=exp01_pretrain
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=sentinel/slurm_logs/exp01_pretrain_%j.out
# exp01 pre-training — submit from project root: sbatch sentinel/experiments/exp01_zero_shot/scripts/slurm/train.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python sentinel/experiments/scripts/run_training.py \
    --config sentinel/experiments/exp01_zero_shot/config.yaml
