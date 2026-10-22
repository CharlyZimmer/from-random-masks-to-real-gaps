#!/usr/bin/env bash
#SBATCH --job-name=exp04_scratch
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=esdc/slurm_logs/exp04_scratch_%j.out
# Training from scratch — submit from project root: sbatch esdc/experiments/exp04_scratch_gapfilling/scripts/slurm/train.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp04_scratch_gapfilling/config.yaml
