# exp01_zero_shot — Max-Gap-Ratio Ablation Results

Sub-experiment of `exp01_zero_shot` (see `../README.md` and this directory's own `README.md`). Same architecture, hyperparameters, and eval dataset as `exp01_zero_shot` — only the pre-training LMDB is post-hoc filtered (at training time, not via a separate dataset) to keep samples whose middle-frame NaN ratio is <= max_gap_ratio, vs. the unfiltered (max_gap_ratio=0.2 at dataset-creation time) baseline. MAE is computed over synthetic gap pixels in the middle frame only, on the same canonical eval set (`$ESDC_GAP_FILLING_DIR/dataset.h5`) used by every other experiment in this repo.

The three tables below additionally break the eval loss down by each evaluation sample's own *natural, pre-donor-injection* gap ratio in the target (bounded <=20% by `max_gap_portion` at eval-set creation time) — independent of the donor-mask-driven gap_mask fraction the loss is computed over. This tests whether the unfiltered model's edge is genuinely from training on gappy data, or just from matching the eval set's own gap-ratio composition: if the gap persists even on the low/zero-natural-gap-ratio subsets, that points to a real effect rather than distribution matching.

## Full evaluation dataset

| ID | max_gap_ratio | Status | Train loss | Val loss | MAE (norm) | MAE (K) | N eval samples | Eval date |
|----|----------------|--------|-----------|---------|-----------|---------|-----------------|-----------|
| exp01_zero_shot_gap000 | 0.0 | done | — | — | 0.0234 | 4.21 | 4569 | 2026-08-05 |
| exp01_zero_shot_gap010 | 0.1 | done | — | — | 0.0228 | 4.10 | 4569 | 2026-08-18 |
| exp01_zero_shot (reference, unfiltered) | 0.2 | done | — | — | 0.0211 | 3.80 | 4569 | 2026-08-18 |

## Evaluation subset: ≤10% natural gap ratio in target

| ID | max_gap_ratio | Status | Train loss | Val loss | MAE (norm) | MAE (K) | N eval samples | Eval date |
|----|----------------|--------|-----------|---------|-----------|---------|-----------------|-----------|
| exp01_zero_shot_gap000 | 0.0 | done | — | — | 0.0233 | 4.20 | 3349 | 2026-08-05 |
| exp01_zero_shot_gap010 | 0.1 | done | — | — | 0.0235 | 4.23 | 3349 | 2026-08-18 |
| exp01_zero_shot (reference, unfiltered) | 0.2 | done | — | — | 0.0214 | 3.85 | 3349 | 2026-08-18 |

## Evaluation subset: 0% natural gap ratio in target (fully clean ground truth)

| ID | max_gap_ratio | Status | Train loss | Val loss | MAE (norm) | MAE (K) | N eval samples | Eval date |
|----|----------------|--------|-----------|---------|-----------|---------|-----------------|-----------|
| exp01_zero_shot_gap000 | 0.0 | done | — | — | 0.0239 | 4.31 | 1281 | 2026-08-05 |
| exp01_zero_shot_gap010 | 0.1 | done | — | — | 0.0261 | 4.69 | 1281 | 2026-08-18 |
| exp01_zero_shot (reference, unfiltered) | 0.2 | done | — | — | 0.0223 | 4.02 | 1281 | 2026-08-18 |

_Regenerate with: `python esdc/experiments/exp01_zero_shot/max_gap_ablation/scripts/collect_results.py` (writes this file directly)._
