#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${S2_DATA_DIR:-./sentinel/data}"

PYTHONPATH=. python sentinel/src/download_data/sentinel_spatiotemporal_downloader.py \
  --ee_project "${EE_PROJECT:?set EE_PROJECT}" \
  --save_path "$SAVE_PATH" \
  --collection "COPERNICUS/S2_SR_HARMONIZED" \
  --dates "2019-01-01" "2025-12-31" \
  --bands "B2" "B3" "B4" "B5" "B6" "B7" "B8" "B8A" "B11" "B12" \
  --dtype "float32" \
  --num_workers 8 \
  --log_freq 1 \
  --match_file "./sentinel/data/ssl4eo-s12_center_coords.csv" \
  --indices_range "0" "50" \
  --patch_size 128 \
  --n_timesteps 5 \
  --max_gap_ratio 0.2 \
  --cld_prb_thresh 50 \
  --nir_drk_thresh 0.15 \
  --cld_prj_dist 1 \
  --buffer 50
