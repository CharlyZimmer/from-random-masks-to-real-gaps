#!/usr/bin/env bash
set -euo pipefail

python sentinel/src/dataset/create_LMDB.py \
  --root ./sentinel/data \
  --subdir SSL-consecutive \
  --lmdb_file ./sentinel/data/ssl4eo_consecutive_s2c.lmdb \
  --num_workers 6 \
  --normalize \
  --value_format raw \
  --test_split 0.1