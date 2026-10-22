# Sentinel-2 Gap-Filling Results

Canonical test set: `$S2_DATA_DIR/gap_filling`  
(HPC: full disjoint-location eval set; local: small subset for pipeline sanity checks only)

MAE is computed over synthetic gap pixels in the middle frame, averaged across all 10 bands.
Values are normalized ([0,1] reflectance) MAE — Sentinel has no single physical unit like ESDC's Kelvin.  
Train / Val loss are the best-epoch training and validation MAE (norm) of the checkpoint used, read from the Lightning logs. "Train N" (training-set size) is entered manually. All three are "—" for the interpolation baselines (no training run).

| ID | Method | Status | Train N | Train loss | Val loss | MAE (norm) | N samples | Eval date |
|----|--------|--------|---------|-----------|---------|-----------|-----------|-----------|
| exp01_zero_shot | Pre-trained MaskFeat model evaluated directly on gap-filling — no fine-tuning | done | — | 0.0098 | 0.0097 | 0.0203 | 10624 | 2026-08-24 |
| exp02_finetuning | Pretrained model fine-tuned on the gap-filling training dataset (fresh optimizer) | pending | — | — | — | — | — | — |
| exp03_structured_masking | Pre-trained MaskFeat model using BEiT-style block masking instead of random masking, evaluated zero-shot on gap-filling | done | — | 0.0133 | 0.0133 | 0.0214 | 10624 | 2026-08-26 |
| exp04_scratch_gapfilling | Training from scratch on the gap-filling dataset (gap_only_masking). Baseline for the learned methods | done | — | 0.0134 | 0.0147 | 0.0210 | 10624 | 2026-09-07 |
| interpolation_baseline | Ordinary Kriging spatial interpolation baseline (no learned model), fit independently per band | done | — | — | — | 0.0380 | 10535 | 2026-08-21 |
| linear_interpolation_baseline | Linear spatial interpolation baseline (no learned model), fit independently per band | done | — | — | — | 0.0386 | 9246 | 2026-08-18 |

_Regenerate with: `python sentinel/experiments/scripts/collect_results.py` (writes this file directly; stdout redirection is not required)_
