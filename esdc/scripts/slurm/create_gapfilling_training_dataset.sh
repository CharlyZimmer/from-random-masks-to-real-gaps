#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=esdc_create_gapfilling_training
#SBATCH --partition=clara
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o esdc/slurm_logs/create_gapfilling_training_dataset_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export ESDC_GAP_FILLING_TRAINING_DIR="${ESDC_GAP_FILLING_TRAINING_DIR:-./esdc/data/lst_gap_filling_training}"
export ESDC_COMPUTED_RANGES_DIR="${ESDC_COMPUTED_RANGES_DIR:-esdc/data}"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python esdc/src/download_data/create_esdc_dataset.py \
  --output_path "${ESDC_GAP_FILLING_TRAINING_DIR}/" \
  --time_start 2002-05-29 \
  --time_end   2010-12-31 \
  --max_gap_portion 0.2 \
  --block_size 5 64 64 \
  --strides 3 32 32 \
  --gap_donor_min_ratio 0.05 \
  --gap_donor_max_ratio 0.85 \
  --syn_gaps \
  --seed 42 \
  --valid_range 180 360 \
  --aux_vars '[{"var": "air_temperature_2m"}]' \
  --computed_ranges_lookup "${ESDC_COMPUTED_RANGES_DIR}/computed_ranges.json"
