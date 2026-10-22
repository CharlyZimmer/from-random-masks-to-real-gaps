#!/usr/bin/env bash
#SBATCH --job-name=exp02_n0100_ft
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=esdc/slurm_logs/exp02_n0100_ft_%j.out
# Sample-efficiency ablation of exp02_finetuning -- 100 samples, 20 epochs.
# Single GPU (unlike n0400-n12800's 4-way DDP): at this scale, 4-way DDP
# leaves too few (or zero) steps/epoch -- see configs/n0100.yaml for the
# devices=1 / base_lr rescaling rationale.
# Submit from project root: sbatch esdc/experiments/exp02_finetuning/sample_ablation/scripts/slurm/train_n0100.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
    --config esdc/experiments/exp02_finetuning/sample_ablation/configs/n0100.yaml \
    --max_samples 100
