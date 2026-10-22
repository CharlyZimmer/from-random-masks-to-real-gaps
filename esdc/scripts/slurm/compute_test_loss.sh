#!/bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=esdc_test_loss
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:00:00
#SBATCH -o esdc/slurm_logs/compute_test_loss_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

PYTHONPATH=. srun python esdc/src/training/compute_test_loss.py \
  --batch_size 32 \
  --num_workers 6 \
  --accelerator cuda \
  --devices 1 \
  --precision 16-mixed
  # --checkpoint /path/to/explicit.ckpt
