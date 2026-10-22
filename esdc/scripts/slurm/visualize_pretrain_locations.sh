#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=esdc_vis_pretrain_loc
#SBATCH --partition=clara
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH -o esdc/slurm_logs/visualize_pretrain_locations_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python esdc/src/visualization/visualize_locations.py \
  --data_dir "${ESDC_DATA_DIR:-./esdc/data/lst_no_syn_gaps}" \
  --out esdc/results/location_distribution/esdc_pretrain_locations.png \
  --title "ESDC Pre-Training Dataset" \
  --vmax 484
