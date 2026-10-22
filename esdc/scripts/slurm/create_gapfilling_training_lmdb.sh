#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=esdc_create_gapfilling_training_lmdb
#SBATCH --partition=clara
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o esdc/slurm_logs/create_gapfilling_training_lmdb_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export ESDC_GAP_FILLING_TRAINING_DIR="${ESDC_GAP_FILLING_TRAINING_DIR:-./esdc/data/lst_gap_filling_training}"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python esdc/src/dataset/create_gapfilling_training_lmdb.py
