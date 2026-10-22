#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=esdc_compute_ranges
#SBATCH --partition=clara
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o esdc/slurm_logs/compute_value_ranges_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export ESDC_COMPUTED_RANGES_DIR="${ESDC_COMPUTED_RANGES_DIR:-esdc/data}"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python esdc/src/analysis/compute_value_ranges.py \
  --vars land_surface_temperature air_temperature_2m \
         leaf_area_index surface_moisture \
  --time_start 2002-05-29 \
  --time_end   2010-12-31 \
  --output "${ESDC_COMPUTED_RANGES_DIR}/computed_ranges.json"
