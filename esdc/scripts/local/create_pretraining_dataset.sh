#!/usr/bin/env bash
set -euo pipefail

export ESDC_DATA_DIR="${ESDC_DATA_DIR:-./esdc/data/lst_no_syn_gaps}"
export ESDC_COMPUTED_RANGES_DIR="${ESDC_COMPUTED_RANGES_DIR:-esdc/data}"

PYTHONPATH=. python esdc/src/download_data/create_esdc_dataset.py \
  --output_path "${ESDC_DATA_DIR}/" \
  --time_start 2003-01-01 \
  --time_end   2003-05-01 \
  --max_gap_portion 0.2 \
  --block_size 5 64 64 \
  --strides 3 32 32 \
  --seed 42 \
  --valid_range 180 360 \
  --aux_vars '[{"var": "air_temperature_2m"}]' \
  --computed_ranges_lookup "${ESDC_COMPUTED_RANGES_DIR}/computed_ranges.json"