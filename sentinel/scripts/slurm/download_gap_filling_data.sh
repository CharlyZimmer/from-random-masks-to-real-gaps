#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=s2_gap_filling_download
#SBATCH --partition=clara
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o sentinel/slurm_logs/download_gap_filling_data_%j.out
#SBATCH --mail-type=END

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

SAVE_PATH="${S2_DATA_DIR:-./sentinel/data}/gap_filling"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python sentinel/src/download_data/sentinel_gapfilling_downloader.py \
  --ee_project "${EE_PROJECT:?set EE_PROJECT}" \
  --ee_service_account "${EE_SERVICE_ACCOUNT:?set EE_SERVICE_ACCOUNT}" \
  --ee_key_file "${EE_KEY_FILE:?set EE_KEY_FILE}" \
  --save_path "$SAVE_PATH" \
  --collection "COPERNICUS/S2_SR_HARMONIZED" \
  --dates "2019-01-01" "2025-12-31" \
  --bands "B2" "B3" "B4" "B5" "B6" "B7" "B8" "B8A" "B11" "B12" \
  --dtype "float32" \
  --num_workers 32 \
  --log_freq 1 \
  --overlap_check "grid" \
  --indices_range 0 30000 \
  --patch_size 128 \
  --n_timesteps 5 \
  --max_gap_ratio 0.2 \
  --gap_donor_min_ratio 0.55 \
  --gap_donor_max_ratio 0.85 \
  --cld_prb_thresh 50 \
  --nir_drk_thresh 0.15 \
  --cld_prj_dist 1 \
  --buffer 50
