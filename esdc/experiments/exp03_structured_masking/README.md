# exp03 — Structured Masking

## What this experiment tests

Same setup as `exp01_zero_shot` (same pre-training data, model size, and
zero-shot gap-filling evaluation) — the only difference is the *shape* of the
masking strategy applied during pre-training:

- `exp01_zero_shot` scatters the random mask across individual patches
  (standard MaskFeat masking).
- `exp03_structured_masking` masks a single contiguous rectangular block
  instead, following the block masking scheme from BEiT (Bao et al., *BEiT:
  BERT Pre-Training of Image Transformers*).

Both experiments aim to mask a similar total budget of the middle frame's
patch grid — exp01's final mask ratio is (by construction) exactly
`Uniform(0.7, 1.0)` — OR'd with the deterministic gap mask, so that masking
*shape* is the variable being isolated to test whether structure-matching
masks (closer to how real cloud gaps look) improve zero-shot gap-filling
performance over scattered random masking. Matching that budget for a single
contiguous block isn't as direct as it sounds — see "Calibrating the block
size range" below.

## Masking algorithm

Implemented in `ViViT.block_masking()` (`model_gap_aware/vivit.py`), enabled
via `structured_masking=True`. Only the middle frame is affected; all other
frames keep the pure deterministic gap mask, same as `random_masking()`.

Per sample:
1. Sample a target block area ratio `size_ratio ~ Uniform(*block_size_range)`
   of the `L`-patch grid, where `block_size_range` defaults to `(0.67, 0.97)`
   — see "Calibrating the block size range" below for why it isn't `(0.7, 1.0)`.
2. Sample an aspect ratio log-uniformly, but **restricted to the sub-range of
   `(0.3, 1/0.3)` that is geometrically feasible for `size_ratio`** —
   `[max(0.3, size_ratio), min(1/0.3, 1/size_ratio)]`. On a square grid, a
   rectangle can only reach area ratio `s` while holding aspect ratio `a` if
   `min(a, 1/a) >= s` (one side is capped by the grid edge), so an aspect
   ratio far from 1 can *never* reach a high area target — e.g. aspect 0.3 or
   3.33 tops out at 30% coverage no matter how it's placed. Without this
   restriction, ~55% of draws undershot the 0.7–1.0 target (confirmed by
   simulation); as `size_ratio` approaches 1.0 the feasible aspect range
   collapses toward 1 (a near-square block), which is the only shape that can
   cover nearly the whole grid.
3. Compute `h = round(sqrt(target_area * aspect))`, `w = round(sqrt(target_area / aspect))`,
   then scale both dimensions down together (not independently) if either
   would exceed the grid, preserving the sampled aspect ratio rather than
   silently distorting it.
4. Place the `h x w` block at a random valid top-left offset in the grid.
5. OR the block with the deterministic gap mask.

With this restriction, actual coverage lands within a percentage point or two of
`size_ratio` (residual is just integer-patch rounding, more visible on
coarser grids) — verified via simulation and against the local sanity-check
checkpoint. The tradeoff is that block shape diversity
across the full `0.3`–`3.33` aspect range is mostly only realized for
lower `size_ratio` draws (~0.7); as the sampled size approaches 1.0 the block
is necessarily close to square.

**Novel-patch check**: patches already masked by the gap mask contribute no
loss signal (only patches masked *and* not already a gap are useful, per
`MaskFeatGapFiller.mae`/`mse`). If a placement's patches-not-already-gapped
("novel") count is below 10% of the grid, the block is resampled (size,
aspect, position) up to 10 tries total — BEiT's own retry convention. The
attempt with the highest novel count is kept if none clears the bar.

When `gap_only_masking=True` (used by `evaluate_gapfilling.py` for zero-shot
inference), block masking is bypassed entirely and only the deterministic gap
mask is applied — identical behavior to `random_masking`'s `gap_only_masking`
path, so evaluation works exactly as for exp01.

## Calibrating the block size range

`block_masking` sizes its rectangle independently of any pre-existing gap
patches, then OR's it with the deterministic gap mask — so the *final*
mid-frame mask ratio is always `>= size_ratio` and typically larger whenever
the block doesn't fully swallow the pre-existing gaps. `random_masking`
doesn't have this problem: it computes the total masked budget first and
fills it including existing gap patches "for free" (they're pushed to the
tail of the sort, so they never push the total *above* the sampled budget) —
which is why exp01's actual mid-frame ratio is (almost) exactly
`Uniform(0.7, 1.0)` (verified against real data: mean 0.854, std 0.086, 0%
below 0.70, only 1.3% exactly at 1.0). With `block_size_range = (0.7, 1.0)`,
the union was skewed well above that — many samples landed exactly at 100%.

`block_size_range`'s bounds (both lower *and* upper) are calibrated to
counteract this, using `esdc/src/analysis/calibrate_block_masking.py`:

- Real middle-frame gap patterns were extracted from
  `$ESDC_DATA_DIR/lmdb/dataset.lmdb` (mean gap coverage 10.5%, median 6.6%,
  23.5% of samples with zero gaps).
- `random_masking`'s ratio distribution on those same real patterns is the
  empirical target.
- A grid search over `block_size_range` (lower, upper) pairs, scored by the
  Kolmogorov-Smirnov statistic against the target (averaged over 3
  independent data subsamples for robustness), found **`(0.67, 0.97)`** as
  the best fit — reducing the mean KS statistic from 0.127 (at (0.7, 1.0)) to
  0.073, matching exp01's mean (0.850 vs. 0.854) and std (0.086 vs. 0.086)
  almost exactly.
- A lower-bound-only search (upper fixed at 1.0) was tried first — best fit
  0.64, mean KS 0.113 — but jointly tuning both bounds fits meaningfully
  better, which is why `block_size_range`'s upper bound is also lowered.
- A per-sample "gap-aware" alternative was tried and rejected: draw
  `target ~ Uniform(0.7, 1.0)` per sample and size the block as
  `target - that_sample's_gap_fraction`. This performed *worse* (KS ≈ 0.247)
  because it assumes the block and the real gap don't overlap, but a large
  randomly-placed block usually fully contains the (typically small) real
  gap, so the "credit" for existing coverage often doesn't materialize,
  causing systematic undershoot instead. Fixing that properly would require
  deliberately biasing block *placement* toward the existing gap region, not
  just a size formula — a bigger algorithmic change, left out of scope here.

**Residual mismatch**: even at the best-fit range, block masking's
distribution isn't identical in shape to exp01's — a somewhat fatter spike at
exactly 100% coverage and a slightly longer left tail below 0.70 persist.
This is structural: a single contiguous rectangle, quantized to whole
patches and randomly placed, can't exactly reproduce exp01's per-patch
budget-fill. Re-run `esdc/src/analysis/calibrate_block_masking.py` if the
pretraining dataset's gap statistics change materially (e.g. a new data
cube, different aux variables, or a different `patch_size`).

## Where it is implemented

- `model_gap_aware/vivit.py` — `ViViT.block_masking()`; the `structured_masking`
  flag selects between `random_masking` and `block_masking` in `forward()`.
- `model_gap_aware/maskfeat.py` — `MaskFeatGapFiller` passes
  `structured_masking`, `block_size_range`, `block_aspect_ratio_range` to `ViViT`.
- `model_gap_aware/lmdb_training.py` — `build_model()` passes
  `structured_masking` through.
- `esdc/src/training/gap_aware_training.py` — `--structured_masking` CLI flag.
- `esdc/experiments/scripts/run_training.py` — reads `structured_masking`
  from `config.yaml`'s `training:` block.
- `esdc/src/analysis/calibrate_block_masking.py` — calibration script for
  `block_size_range` (see "Calibrating the block size range").

## Notes

- Trains from scratch on the same pre-training LMDB as `exp01_zero_shot`
  (`$ESDC_DATA_DIR`), not the gap-filling training set — this experiment is a
  masking-strategy ablation of exp01, not a fine-tuning/scratch-gapfilling
  experiment (compare `exp02_finetuning` / `exp04_scratch_gapfilling`).
- The eval procedure is identical to `exp01_zero_shot` (same
  `evaluate_gapfilling.py`).
