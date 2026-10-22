#!/bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=s2_gapfilling_eval
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:00:00
#SBATCH -o sentinel/slurm_logs/evaluate_gapfilling_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

DATA_DIR="${S2_DATA_DIR:-./sentinel/data}/gap_filling"

PYTHONPATH=. srun python sentinel/src/evaluation/evaluate_gapfilling.py \
  --data_dir "$DATA_DIR" \
  --log_dir  ./sentinel/lightning_logs/pretraining \
  --device   cuda \
  --batch_size 16 \
  --normalize
  # --checkpoint /path/to/explicit.ckpt
  # --num_samples 1000
