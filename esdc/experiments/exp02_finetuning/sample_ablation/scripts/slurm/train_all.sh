#!/usr/bin/env bash
#SBATCH --job-name=exp02_ft_ablation_all
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=esdc/slurm_logs/exp02_ft_ablation_all_%j.out
# Sample-efficiency ablation of exp02_finetuning -- all 6 sample counts
# (400/800/1600/3200/6400/12800, 20 epochs each) run as sequential job steps
# inside a SINGLE resource allocation, so the 4-GPU reservation is only
# queued for once instead of 6 separate times (see train_n*.sh for the
# equivalent one-variant-per-job scripts, useful for re-running a single
# variant in isolation).
#
# --time is the sum of the individual train_n*.sh budgets (4+4+4+4+6+10=32h,
# themselves rough linear estimates -- see sample_ablation/README.md) plus a
# buffer for 6x job-step startup overhead.
#
# Submit from project root: sbatch esdc/experiments/exp02_finetuning/sample_ablation/scripts/slurm/train_all.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

CFG_DIR="esdc/experiments/exp02_finetuning/sample_ablation/configs"

declare -A SAMPLES=( [n0400]=400 [n0800]=800 [n1600]=1600 [n3200]=3200 [n6400]=6400 [n12800]=12800 )

for label in n0400 n0800 n1600 n3200 n6400 n12800; do
    echo "=== sample_ablation train_all: ${label} (${SAMPLES[$label]} samples) ==="
    PYTHONPATH=. srun python esdc/experiments/scripts/run_training.py \
        --config "${CFG_DIR}/${label}.yaml" \
        --max_samples "${SAMPLES[$label]}"
done
