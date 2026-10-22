#!/usr/bin/env bash
#SBATCH --job-name=exp02_n12800_ft
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0-10:00:00
#SBATCH --output=esdc/slurm_logs/exp02_n12800_ft_%j.out
# Sample-efficiency ablation of exp02_finetuning -- 12800 samples, 20 epochs.
# --time is a rough linear estimate scaled from n6400's own budget (total_steps
# 3800 vs. n6400's 1900, see sample_ablation/README.md).
# Submit from project root: sbatch esdc/experiments/exp02_finetuning/sample_ablation/scripts/slurm/train_n12800.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp02_finetuning/sample_ablation/configs/n12800.yaml \
    --max_samples 12800
