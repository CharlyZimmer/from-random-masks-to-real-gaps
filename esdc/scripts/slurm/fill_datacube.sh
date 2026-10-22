#!/bin/bash

#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=esdc_fill_datacube
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=2-00:00:00
#SBATCH -o esdc/slurm_logs/fill_datacube_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

DATA_DIR="${ESDC_DATA_DIR:-./esdc/data/lst_no_syn_gaps}"

PYTHONPATH=. srun python esdc/src/datacube_gapfilling/fill_datacube.py \
  --data_dir  "$DATA_DIR" \
  --log_dir   ./esdc/lightning_logs/pretraining \
  --time_span 2002-05-29:2011-12-23 \
  --device    cuda \
  --batch_size 64
  # --checkpoint /path/to/explicit.ckpt
