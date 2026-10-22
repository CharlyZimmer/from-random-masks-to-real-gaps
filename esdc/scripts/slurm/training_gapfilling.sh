#!/bin/bash

#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=esdc_training_gapfilling
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --time=2-00:00:00
#SBATCH -o esdc/slurm_logs/training_gapfilling_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export ESDC_DATA_DIR="${ESDC_GAP_FILLING_TRAINING_DIR:-./esdc/data/lst_gap_filling_training}"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

NUM_GPUS=4
BASE_LR=1e-4
LR=$(python -c "print(${BASE_LR} * ${NUM_GPUS})")

srun python esdc/src/training/gap_aware_training.py \
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
  --precision 16-mixed \
  --gap_only_masking \
  --y_lmdb_path "${ESDC_DATA_DIR}/lmdb/dataset_y.lmdb" \
  --log_dir esdc/lightning_logs/gapfilling_training
