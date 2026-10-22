#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python sentinel/src/visualization/visualize_locations.py \
  --data_dir "${S2_DATA_DIR:-./sentinel/data}/gap_filling" \
  --out sentinel/results/location_distribution/sentinel_gapfilling_locations.png \
  --title "Sentinel Gap-Filling Dataset"
