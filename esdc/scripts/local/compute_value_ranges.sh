#!/usr/bin/env bash
set -euo pipefail

export ESDC_COMPUTED_RANGES_DIR="${ESDC_COMPUTED_RANGES_DIR:-esdc/data}"

PYTHONPATH=. python esdc/src/analysis/compute_value_ranges.py \
  --vars land_surface_temperature air_temperature_2m \
         leaf_area_index surface_moisture \
  --time_start 2003-01-01 \
  --time_end   2003-05-01 \
  --output "${ESDC_COMPUTED_RANGES_DIR}/computed_ranges.json"
