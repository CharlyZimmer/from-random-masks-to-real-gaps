#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=s2_vis_gapfill_loc
#SBATCH --partition=clara
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH -o sentinel/slurm_logs/visualize_gapfilling_locations_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python sentinel/src/visualization/visualize_locations.py \
  --data_dir "${S2_DATA_DIR:-./sentinel/data}/gap_filling" \
  --out sentinel/results/location_distribution/sentinel_gapfilling_locations.png \
  --title "Sentinel Gap-Filling Dataset"
