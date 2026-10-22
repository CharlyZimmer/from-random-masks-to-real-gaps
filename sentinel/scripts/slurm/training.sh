#!/bin/bash

#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=s2_training
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:2
#SBATCH --time=2-00:00:00
#SBATCH -o sentinel/slurm_logs/training_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

NUM_GPUS=2
BASE_LR=1e-4
LR=$(python -c "print(${BASE_LR} * ${NUM_GPUS})")

LMDB_PATH="${S2_LMDB_PATH:-./sentinel/data/ssl4eo_consecutive_s2c.lmdb}"

PYTHONPATH=. srun python sentinel/src/training/gap_aware_training.py \
  --lmdb_path "$LMDB_PATH" \
  --patch_size 4 \
  --batch_size 16 \
  --num_workers 6 \
  --embed_dim 384 \
  --depth 8 \
  --num_heads 6 \
  --decoder_heads 1 \
  --lr $LR \
  --max_epochs 50 \
  --warmup_steps 10000 \
  --accelerator cuda \
  --devices $NUM_GPUS \
  --precision 16-mixed
