# ESDC Gap-Filling Results

Canonical test set: `$ESDC_GAP_FILLING_DIR/dataset.h5`  
(HPC: last year of ESDC LST coverage, temporally disjoint from training;  
local: 2004 data for pipeline sanity checks only)

MAE is computed over synthetic gap pixels in the middle frame only.  
Train / Val loss are the best-epoch training and validation MAE (norm) of the checkpoint used, read from the Lightning logs. "Train N" (training-set size) is entered manually. All three are "—" for the interpolation baselines (no training run).

| ID | Method | Status | Train N | Train loss | Val loss | MAE (norm) | MAE (K) | N samples | Eval date |
|----|--------|--------|---------|-----------|---------|-----------|---------|-----------|-----------|
| exp01_zero_shot | Pre-trained MaskFeat model evaluated directly on gap-filling — no fine-tuning | done | — | 0.0097 | 0.0096 | 0.0211 | 3.80 | 4569 | 2026-08-18 |
| exp02_finetuning | Pretrained model fine-tuned on the gap-filling training dataset (fresh optimizer) | done | — | 0.0118 | 0.0101 | 0.0184 | 3.31 | 4569 | 2026-08-03 |
| exp03_structured_masking | Pre-trained MaskFeat model using BEiT-style block masking instead of random masking, evaluated zero-shot on gap-filling | done | — | 0.0128 | 0.0127 | 0.0180 | 3.24 | 4569 | 2026-08-26 |
| exp04_scratch_gapfilling | Training from scratch on the gap-filling dataset (gap_only_masking). Baseline for ablations | done | — | 0.0124 | 0.0142 | 0.0173 | 3.11 | 4569 | 2026-08-26 |
| interpolation_baseline | Ordinary Kriging spatial interpolation baseline (no learned model, no aux variable) | done | — | — | — | 0.0195 | 3.51 | 3614 | 2026-07-24 |
| linear_interpolation_baseline | Linear spatial interpolation baseline (no learned model, no aux variable) | done | — | — | — | 0.0213 | 3.84 | 3614 | 2026-08-03 |

_Regenerate with: `python esdc/experiments/scripts/collect_results.py` (writes this file directly)_
