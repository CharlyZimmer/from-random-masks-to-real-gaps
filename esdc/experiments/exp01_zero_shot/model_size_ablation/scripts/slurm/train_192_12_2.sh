#!/usr/bin/env bash
#SBATCH --job-name=exp01_192_12_2_pretrain
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=esdc/slurm_logs/exp01_192_12_2_pretrain_%j.out
# model_size_ablation (192/12/2) pre-training — submit from project root:
# sbatch esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/slurm/train_192_12_2.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp01_zero_shot/model_size_ablation/configs/192_12_2.yaml
