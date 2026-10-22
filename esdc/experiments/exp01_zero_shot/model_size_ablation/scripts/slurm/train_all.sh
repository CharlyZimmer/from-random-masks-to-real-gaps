#!/usr/bin/env bash
#SBATCH --job-name=exp01_model_size_ablation_all
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=esdc/slurm_logs/exp01_model_size_ablation_all_%j.out
# Model-size ablation of exp01_zero_shot -- all 6 smaller ViT configs
# (192/12/2, 192/6/2, 96/6/2, 96/6/1, 96/3/1, 48/3/1; 50 epochs each, same
# pre-training LMDB and hyperparameters as exp01_zero_shot itself) run as
# sequential job steps inside a SINGLE resource allocation, so the 4-GPU
# reservation is only queued for once instead of 6 separate times (see
# train_<label>.sh for the equivalent one-variant-per-job scripts, useful for
# re-running a single variant in isolation).
#
# The (192,12,3) baseline is NOT retrained here -- it's identical to
# exp01_zero_shot's own default architecture, whose checkpoint/results are
# reused directly (see ../../README.md and scripts/collect_results.py).
#
# Submit from project root: sbatch esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/slurm/train_all.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

CFG_DIR="esdc/experiments/exp01_zero_shot/model_size_ablation/configs"

for label in 192_12_2 192_6_2 96_6_2 96_6_1 96_3_1 48_3_1; do
    echo "=== model_size_ablation train_all: ${label} ==="
    PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
        --config "${CFG_DIR}/${label}.yaml"
done
