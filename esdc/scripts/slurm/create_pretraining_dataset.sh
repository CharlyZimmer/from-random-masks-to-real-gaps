#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=esdc_create_pretraining
#SBATCH --partition=clara
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o esdc/slurm_logs/create_pretraining_dataset_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

export ESDC_DATA_DIR="${ESDC_DATA_DIR:-./esdc/data/lst_no_syn_gaps}"
export ESDC_COMPUTED_RANGES_DIR="${ESDC_COMPUTED_RANGES_DIR:-esdc/data}"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python esdc/src/download_data/create_esdc_dataset.py \
  --output_path "${ESDC_DATA_DIR}/" \
  --time_start 2002-05-29 \
  --time_end   2010-12-31 \
  --max_gap_portion 0.2 \
  --seed 42 \
  --valid_range 180 360 \
  --strides 3 32 32 \
  --block_size 5 64 64 \
  --aux_vars '[{"var": "air_temperature_2m"}]' \
  --computed_ranges_lookup "${ESDC_COMPUTED_RANGES_DIR}/computed_ranges.json"
