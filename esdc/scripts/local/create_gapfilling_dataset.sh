#!/usr/bin/env bash
set -euo pipefail

export ESDC_GAP_FILLING_DIR="${ESDC_GAP_FILLING_DIR:-./esdc/data/lst_gap_filling}"
export ESDC_COMPUTED_RANGES_DIR="${ESDC_COMPUTED_RANGES_DIR:-esdc/data}"

PYTHONPATH=. python esdc/src/download_data/create_esdc_dataset.py \
  --output_path "${ESDC_GAP_FILLING_DIR}/" \
  --time_start 2004-01-01 \
  --time_end   2004-06-01 \
  --max_gap_portion 0.2 \
  --block_size 5 64 64 \
  --strides 3 32 32 \
  --gap_donor_min_ratio 0.05 \
  --gap_donor_max_ratio 0.85 \
  --syn_gaps \
  --no_split \
  --valid_range 180 360 \
  --aux_vars '[{"var": "air_temperature_2m"}]' \
  --computed_ranges_lookup "${ESDC_COMPUTED_RANGES_DIR}/computed_ranges.json"
