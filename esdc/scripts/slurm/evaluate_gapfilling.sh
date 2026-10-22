#!/bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=esdc_gapfilling_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:00:00
#SBATCH -o esdc/slurm_logs/evaluate_gapfilling_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

DATA_H5="${ESDC_GAP_FILLING_DIR:-./esdc/data/lst_gap_filling}/dataset.h5"

PYTHONPATH=. srun python esdc/src/evaluation/evaluate_gapfilling.py \
  --data_dir  "$DATA_H5" \
  --log_dir   ./esdc/lightning_logs/pretraining \
  --device    cuda \
  --batch_size 16
  # --checkpoint /path/to/explicit.ckpt
  # --num_samples 1000
