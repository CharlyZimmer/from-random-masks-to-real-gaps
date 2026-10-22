#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=esdc_vis_gapfill_loc
#SBATCH --partition=clara
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH -o esdc/slurm_logs/visualize_gapfilling_locations_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python esdc/src/visualization/visualize_locations.py \
  --data_dir "${ESDC_GAP_FILLING_DIR:-./esdc/data/lst_gap_filling}" \
  --out esdc/results/location_distribution/esdc_gapfilling_locations.png \
  --title "ESDC Gap-Filling Dataset"
