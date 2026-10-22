#!/usr/bin/env bash
#SBATCH --job-name=exp01_gap000_pretrain
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=esdc/slurm_logs/exp01_gap000_pretrain_%j.out
# max_gap_ablation (0%) pre-training — submit from project root:
# sbatch esdc/experiments/exp01_zero_shot/max_gap_ablation/scripts/slurm/train_gap000.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp01_zero_shot/max_gap_ablation/configs/gap000.yaml
