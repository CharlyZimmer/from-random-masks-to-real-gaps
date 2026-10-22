#!/usr/bin/env bash
#SBATCH --job-name=exp03_pretrain
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=esdc/slurm_logs/exp03_pretrain_%j.out
# exp03 pre-training (BEiT-style block masking) — submit from project root: sbatch esdc/experiments/exp03_structured_masking/scripts/slurm/train.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp03_structured_masking/config.yaml
