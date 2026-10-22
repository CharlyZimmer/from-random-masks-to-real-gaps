#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python sentinel/src/visualization/visualize_locations.py \
  --data_dir "${S2_DATA_DIR:-./sentinel/data}" \
  --out sentinel/results/location_distribution/sentinel_pretrain_locations.png \
  --title "Sentinel Pre-Training Dataset"
