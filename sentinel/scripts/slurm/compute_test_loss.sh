#!/bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=s2_test_loss
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:00:00
#SBATCH -o sentinel/slurm_logs/compute_test_loss_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

LMDB_PATH="${S2_LMDB_PATH:-./sentinel/data/ssl4eo_consecutive_s2c.lmdb}"
TEST_LMDB="${LMDB_PATH%.lmdb}_test.lmdb"

PYTHONPATH=. srun python sentinel/src/training/compute_test_loss.py \
  --test_lmdb_path "$TEST_LMDB" \
  --batch_size 16 \
  --num_workers 6 \
  --accelerator cuda \
  --devices 1 \
  --precision 16-mixed
  # --checkpoint /path/to/explicit.ckpt
