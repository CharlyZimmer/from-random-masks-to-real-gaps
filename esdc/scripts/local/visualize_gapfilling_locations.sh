#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python esdc/src/visualization/visualize_locations.py \
  --data_dir "${ESDC_GAP_FILLING_DIR:-./esdc/data/lst_gap_filling}" \
  --out esdc/results/location_distribution/esdc_gapfilling_locations.png \
  --title "ESDC Gap-Filling Dataset"
