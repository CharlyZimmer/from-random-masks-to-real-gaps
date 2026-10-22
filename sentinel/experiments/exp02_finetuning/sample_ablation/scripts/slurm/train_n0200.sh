#!/usr/bin/env bash
#SBATCH --job-name=exp02_n0200_ft
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=sentinel/slurm_logs/exp02_n0200_ft_%j.out
# Sample-efficiency ablation of exp02_finetuning -- 200 samples, 20 epochs.
# Single GPU (unlike n0400-n12800's 4-way DDP): at this scale, 4-way DDP
# leaves too few steps/epoch per replica -- see configs/n0200.yaml for the
# devices=1 / base_lr rescaling rationale.
# Submit from project root: sbatch sentinel/experiments/exp02_finetuning/sample_ablation/scripts/slurm/train_n0200.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. python sentinel/experiments/scripts/run_training.py \
    --config sentinel/experiments/exp02_finetuning/sample_ablation/configs/n0200.yaml \
    --max_samples 200
