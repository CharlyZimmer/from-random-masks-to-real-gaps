#!/usr/bin/env bash
#SBATCH --job-name=exp02_finetune
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=sentinel/slurm_logs/exp02_finetune_%j.out
# Fine-tuning — submit from project root: sbatch sentinel/experiments/exp02_finetuning/scripts/slurm/train.sh
# Single GPU -- see config.yaml for the devices=1/base_lr/warmup_steps rationale.
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python sentinel/experiments/scripts/run_training.py \
    --config sentinel/experiments/exp02_finetuning/config.yaml
