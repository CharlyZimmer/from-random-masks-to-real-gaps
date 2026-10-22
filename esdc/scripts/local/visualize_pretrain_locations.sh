#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python esdc/src/visualization/visualize_locations.py \
  --data_dir "${ESDC_DATA_DIR:-./esdc/data/lst_no_syn_gaps}" \
  --out esdc/results/location_distribution/esdc_pretrain_locations.png \
  --title "ESDC Pre-Training Dataset"
