#!/usr/bin/env bash
set -euo pipefail

export ESDC_GAP_FILLING_TRAINING_DIR="${ESDC_GAP_FILLING_TRAINING_DIR:-./esdc/data/lst_gap_filling_training}"

PYTHONPATH=. python esdc/src/dataset/create_gapfilling_training_lmdb.py
