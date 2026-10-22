#!/usr/bin/env bash
# Build the lean gap-mask LMDB from the (small, local) gap-filling training
# directory. Run from project root, after download_gap_filling_training_data.sh:
#   ./sentinel/scripts/local/create_gapfilling_training_lmdb.sh
#
# Only gap masks are stored/read here -- the gappy input series and the
# ground-truth middle frame are reconstructed at training time from the
# pretraining LMDB + these masks (see model_gap_aware/lmdb_training.py::
# PretrainGapMaskJoinDataset), so no --normalize flag is needed here.
set -euo pipefail

ROOT="${S2_GAP_FILLING_TRAINING_DIR:-./sentinel/data/gap_filling_training}"

PYTHONPATH=. python sentinel/src/dataset/create_gap_mask_lmdb.py \
  --root "$ROOT" \
  --test_split 0.2 \
  --num_workers 2
