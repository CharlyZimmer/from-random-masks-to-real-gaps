#!/usr/bin/env bash
#SBATCH --job-name=exp02_finetune
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=esdc/slurm_logs/exp02_finetune_%j.out
# Fine-tuning — submit from project root: sbatch esdc/experiments/exp02_finetuning/scripts/slurm/train.sh
# Single GPU, N=100 samples -- see config.yaml for the devices=1/base_lr/warmup_steps rationale.
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp02_finetuning/config.yaml \
    --max_samples 100
