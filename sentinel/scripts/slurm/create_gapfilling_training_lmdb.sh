#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=s2_create_gap_mask_lmdb
#SBATCH --partition=clara
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH -o sentinel/slurm_logs/create_gap_mask_lmdb_%j.out
#SBATCH --mail-type=END

# Builds the lean gap-mask LMDB from the gap-filling training directory. Only
# gap masks are stored/read here -- the gappy input series and the
# ground-truth middle frame are reconstructed at training time from the
# pretraining LMDB + these masks (see model_gap_aware/lmdb_training.py::
# PretrainGapMaskJoinDataset), so no --normalize flag is needed here.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

ROOT="${S2_GAP_FILLING_TRAINING_DIR:-./sentinel/data/gap_filling_training}"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python sentinel/src/dataset/create_gap_mask_lmdb.py \
  --root "$ROOT" \
  --test_split 0.2 \
  --num_workers 8
