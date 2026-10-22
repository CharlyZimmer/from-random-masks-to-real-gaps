#!/usr/bin/env bash
# Download synthetic gap masks for a small Sentinel-2 gap-filling *training*
# dataset, for local pipeline verification (exp02_finetuning /
# exp04_scratch_gapfilling).
#
# Reuses the pretraining dataset's already-downloaded imagery for the same
# SSL4EO-S12 locations (--pretrain_dir) instead of re-downloading a second
# copy -- only the small synthetic gap mask is new information (see
# sentinel_gapfilling_training_downloader.py). Requires ./sentinel/scripts/local/download_data.sh
# to have already been run for these indices. Only the canonical gap-filling
# eval set (${S2_DATA_DIR}/gap_filling, Gaussian-sampled) stays a fully
# independent, spatially-disjoint download from pre-training.
#
# Output goes to $S2_GAP_FILLING_TRAINING_DIR (or ./sentinel/data/gap_filling_training).
# Run from the project root:
#   ./sentinel/scripts/local/download_gap_filling_training_data.sh
set -euo pipefail

SAVE_PATH="${S2_GAP_FILLING_TRAINING_DIR:-./sentinel/data/gap_filling_training}"
PRETRAIN_DIR="${S2_DATA_DIR:-./sentinel/data}"

PYTHONPATH=. python sentinel/src/download_data/sentinel_gapfilling_training_downloader.py \
  --ee_project "${EE_PROJECT:?set EE_PROJECT}" \
  --save_path "$SAVE_PATH" \
  --pretrain_dir "$PRETRAIN_DIR" \
  --collection "COPERNICUS/S2_SR_HARMONIZED" \
  --bands "B2" "B3" "B4" "B5" "B6" "B7" "B8" "B8A" "B11" "B12" \
  --dtype "float32" \
  --num_workers 8 \
  --log_freq 1 \
  --indices_range "0" "15" \
  --patch_size 128 \
  --max_gap_ratio 0.2 \
  --gap_donor_min_ratio 0.05 \
  --gap_donor_max_ratio 0.85 \
  --cld_prb_thresh 50 \
  --nir_drk_thresh 0.15 \
  --cld_prj_dist 1 \
  --buffer 50 \
  --window_pad_days 15 \
  --donor_window_expand_days 60 \
  --donor_probe_k 20 \
  --resume "$SAVE_PATH/checked_locations.csv"

# --resume keeps gap_masks/{idx}/ already written and only re-attempts skipped
# indices. For a clean rebuild, delete "$SAVE_PATH" first.
