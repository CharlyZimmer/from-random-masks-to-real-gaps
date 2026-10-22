# exp01_zero_shot — Max-Gap-Ratio Ablation Results (Sentinel)

Sub-experiment of `exp01_zero_shot` (see `../README.md` and this directory's own `README.md`). Same architecture, hyperparameters, and eval dataset as `exp01_zero_shot` — only the pre-training LMDB (`$S2_LMDB_PATH`) is post-hoc filtered (at training time, not via a separate dataset) to keep samples whose middle-frame band-0 NaN ratio is <= max_gap_ratio, vs. the pre-training LMDB's own 0.2 creation-time cap (`exp01_zero_shot`, effectively unfiltered relative to this ablation). MAE is computed over synthetic gap pixels in the middle frame, averaged across all 10 bands, on the same canonical eval set (`$S2_DATA_DIR/gap_filling`) used by every other experiment in this repo. Sentinel has no single physical unit like ESDC's Kelvin, so only normalized ([0,1] reflectance) MAE is shown, and (unlike ESDC's version of this ablation) there is no natural-gap-ratio subset breakdown -- sentinel's evaluate_gapfilling.py doesn't compute one.

| ID | max_gap_ratio | Status | Train loss | Val loss | MAE (norm) | N eval samples | Eval date |
|----|----------------|--------|-----------|---------|-----------|-----------------|-----------|
| exp01_zero_shot_gap000 | 0.0 | pending | — | — | — | — | — |
| exp01_zero_shot_gap010 | 0.1 | pending | — | — | — | — | — |
| exp01_zero_shot (reference, unfiltered) | 0.2 | done | — | — | 0.0203 | 10624 | 2026-08-24 |

_Regenerate with: `python sentinel/experiments/exp01_zero_shot/max_gap_ablation/scripts/collect_results.py` (writes this file directly)._
