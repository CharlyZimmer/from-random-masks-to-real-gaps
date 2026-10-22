# exp01_zero_shot — Model-Size Ablation Results

Sub-experiment of `exp01_zero_shot` (see `../README.md` and this directory's own `README.md`). Same pre-training LMDB, hyperparameters, and eval dataset as `exp01_zero_shot` — only the ViT architecture (embed_dim, depth, num_heads) is shrunk. MAE is computed over synthetic gap pixels in the middle frame only, on the same canonical eval set (`$ESDC_GAP_FILLING_DIR/dataset.h5`) used by every other experiment in this repo.

Unlike `max_gap_ablation`, this ablation studies MAE vs. model capacity rather than vs. gap-ratio composition, so evaluation is restricted to the full evaluation dataset (`evaluate_gapfilling.py --full_only`) — no breakdown by natural gap ratio.

## Full evaluation dataset

| ID | Architecture (embed/depth/heads) | Params | Status | Train loss | Val loss | MAE (norm) | MAE (K) | N eval samples | Eval date |
|----|-----------------------------------|--------|--------|-----------|---------|-----------|---------|-----------------|-----------|
| exp01_zero_shot (baseline, reused) | 192/12/3 | 11,105,488 | done | — | — | 0.0211 | 3.80 | 4569 | 2026-08-18 |
| exp01_zero_shot_192_12_2 | 192/12/2 | 9,925,840 | done | — | — | 0.0210 | 3.78 | 4569 | 2026-08-18 |
| exp01_zero_shot_192_6_2 | 192/6/2 | 5,184,208 | done | — | — | 0.0263 | 4.74 | 4569 | 2026-08-18 |
| exp01_zero_shot_96_6_2 | 96/6/2 | 1,661,296 | done | — | — | 0.0228 | 4.11 | 4569 | 2026-08-18 |
| exp01_zero_shot_96_6_1 | 96/6/1 | 1,366,384 | done | — | — | 0.0241 | 4.34 | 4569 | 2026-08-18 |
| exp01_zero_shot_96_3_1 | 96/3/1 | 770,800 | done | — | — | 0.0242 | 4.36 | 4569 | 2026-08-18 |
| exp01_zero_shot_48_3_1 | 48/3/1 | 263,296 | done | — | — | 0.0289 | 5.20 | 4569 | 2026-08-18 |

_Regenerate with: `python esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/collect_results.py` (writes this file directly)._
