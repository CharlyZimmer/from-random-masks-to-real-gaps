#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${ESDC_DATA_DIR:-./esdc/data/lst_no_syn_gaps}"

PYTHONPATH=. python esdc/src/datacube_gapfilling/fill_datacube.py \
  --data_dir  "$DATA_DIR" \
  --log_dir   ./esdc/lightning_logs/pretraining \
  --time_span 2004-01-06:2004-01-19 \
  --device    cpu \
  --batch_size 4
