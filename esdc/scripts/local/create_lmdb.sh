#!/usr/bin/env bash
set -euo pipefail

export ESDC_DATA_DIR="${ESDC_DATA_DIR:-./esdc/data/lst_no_syn_gaps}"

python esdc/src/dataset/create_lmdb.py
