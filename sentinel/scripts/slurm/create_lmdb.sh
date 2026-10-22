#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=make_lmdb
#SBATCH --partition=clara
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o sentinel/slurm_logs/create_lmdb_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

ROOT="${S2_DATA_DIR:-./sentinel/data}"
LMDB="${S2_LMDB_PATH:-./sentinel/data/ssl4eo_consecutive_s2c.lmdb}"

python sentinel/src/dataset/create_LMDB.py \
  --root "$ROOT" \
  --subdir SSL-consecutive \
  --lmdb_file "$LMDB" \
  --num_workers 8 \
  --normalize \
  --value_format raw \
  --test_split 0.1 \
