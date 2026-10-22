#!/usr/bin/env bash
#SBATCH --job-name=exp02_ft_ablation_small_all
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=esdc/slurm_logs/exp02_ft_ablation_small_all_%j.out
# Sample-efficiency ablation of exp02_finetuning -- the 3 smallest sample
# counts (50/100/200, 20 epochs each) run as sequential job steps inside a
# SINGLE 1-GPU resource allocation. Separate from train_all.sh (which covers
# n0400-n12800 on 4-way DDP): these 3 configs use devices=1 (see
# configs/n0050.yaml for why 4-way DDP breaks at this scale), so they need a
# 1-GPU allocation, not the 4-GPU one train_all.sh requests -- mixing
# different DDP degrees inside one job's srun steps doesn't work cleanly.
# See train_n0050.sh/train_n0100.sh/train_n0200.sh for the equivalent
# one-variant-per-job scripts, useful for re-running a single variant.
#
# Submit from project root: sbatch esdc/experiments/exp02_finetuning/sample_ablation/scripts/slurm/train_small_all.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

CFG_DIR="esdc/experiments/exp02_finetuning/sample_ablation/configs"

declare -A SAMPLES=( [n0050]=50 [n0100]=100 [n0200]=200 )

for label in n0050 n0100 n0200; do
    echo "=== sample_ablation train_small_all: ${label} (${SAMPLES[$label]} samples) ==="
    PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
        --config "${CFG_DIR}/${label}.yaml" \
        --max_samples "${SAMPLES[$label]}"
done
