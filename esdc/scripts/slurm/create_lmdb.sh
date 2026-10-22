#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=esdc_make_lmdb
#SBATCH --partition=clara
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o esdc/slurm_logs/create_lmdb_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

python esdc/src/dataset/create_lmdb.py
