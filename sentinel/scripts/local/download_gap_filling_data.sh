#!/usr/bin/env bash
# Download Sentinel-2 gap-filling evaluation dataset.
#
# Output goes to $S2_DATA_DIR/gap_filling/ (or ./sentinel/data/gap_filling/).
# Each sample contains:
#   SSL-consecutive/{idx}/  — 5-step series, middle frame has synthetic NaN gaps
#   ground_truth/{idx}/     — clean middle frame + binary gap_mask.tif
#
# Run from the project root:
#   ./sentinel/scripts/local/download_gap_filling_data.sh
set -euo pipefail

SAVE_PATH="${S2_DATA_DIR:-./sentinel/data}/gap_filling"

PYTHONPATH=. python sentinel/src/download_data/sentinel_gapfilling_downloader.py \
  --ee_project "${EE_PROJECT:?set EE_PROJECT}" \
  --save_path "$SAVE_PATH" \
  --collection "COPERNICUS/S2_SR_HARMONIZED" \
  --dates "2019-01-01" "2025-12-31" \
  --bands "B2" "B3" "B4" "B5" "B6" "B7" "B8" "B8A" "B11" "B12" \
  --dtype "float32" \
  --num_workers 8 \
  --log_freq 1 \
  --overlap_check "rtree" \
  --indices_range 0 30 \
  --patch_size 128 \
  --n_timesteps 5 \
  --max_gap_ratio 0.2 \
  --gap_donor_min_ratio 0.05 \
  --gap_donor_max_ratio 0.85 \
  --cld_prb_thresh 50 \
  --nir_drk_thresh 0.15 \
  --cld_prj_dist 1 \
  --buffer 50
