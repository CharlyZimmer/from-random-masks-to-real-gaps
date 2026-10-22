#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=s2_gap_filling_training_download
#SBATCH --partition=clara
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o sentinel/slurm_logs/download_gap_filling_training_data_%j.out
#SBATCH --mail-type=END

# Downloads synthetic gap masks for the Sentinel-2 gap-filling *training*
# dataset used by exp02_finetuning / exp04_scratch_gapfilling. Reuses the
# pretraining dataset's already-downloaded imagery for the same SSL4EO-S12
# locations (--pretrain_dir) instead of re-downloading a second copy -- only
# the small synthetic gap mask is new information (see
# sentinel_gapfilling_training_downloader.py). Requires sentinel/scripts/slurm/download_data.sh to
# have already completed for these indices. Only the canonical gap-filling
# eval set (indices 0-30000, Gaussian-sampled, no --match_file) stays a fully
# independent, spatially-disjoint download from pre-training -- see
# sentinel/scripts/slurm/download_gap_filling_data.sh.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gap-aware

SAVE_PATH="${S2_GAP_FILLING_TRAINING_DIR:-./sentinel/data/gap_filling_training}"
PRETRAIN_DIR="${S2_DATA_DIR:-./sentinel/data}"
export PYTHONPATH="${GAP_AWARE_ROOT:-.}"

python sentinel/src/download_data/sentinel_gapfilling_training_downloader.py \
  --ee_project "${EE_PROJECT:?set EE_PROJECT}" \
  --ee_service_account "${EE_SERVICE_ACCOUNT:?set EE_SERVICE_ACCOUNT}" \
  --ee_key_file "${EE_KEY_FILE:?set EE_KEY_FILE}" \
  --save_path "$SAVE_PATH" \
  --pretrain_dir "$PRETRAIN_DIR" \
  --collection "COPERNICUS/S2_SR_HARMONIZED" \
  --bands "B2" "B3" "B4" "B5" "B6" "B7" "B8" "B8A" "B11" "B12" \
  --dtype "float32" \
  --num_workers 64 \
  --log_freq 1000 \
  --indices_range "0" "251078" \
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

# --resume keeps every gap_masks/{idx}/ already written and only re-attempts
# skipped indices, so recovery passes are cheap. For a clean full rebuild,
# delete "$SAVE_PATH" first, then submit again.
