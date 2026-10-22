# linear_interpolation_baseline

Non-learned baseline: linear spatial interpolation, evaluated on the same
canonical gap-filling dataset as every other experiment
(`$ESDC_GAP_FILLING_DIR/dataset.h5`, see `../eval_dataset.yaml`).

## Method

For each sample, interpolates the synthetic gap pixels of the middle frame
from the observed (non-NaN) LST pixels via piecewise-linear interpolation
over their Delaunay triangulation (`scipy.interpolate.griddata`,
`method="linear"`). Target points outside the convex hull of the observed
pixels (which happens when a gap patch touches the image border/corner) are
filled in with nearest-neighbor interpolation as a fallback.

Same scope restriction as `../interpolation_baseline` (Kriging), to keep the
two interpolation baselines directly comparable:
- **Auxiliary variable** (`air_temperature_2m`) — not used.
- **Temporal context** — the other 4 frames of the 5-frame input series are
  ignored; only the middle frame's spatial neighbors inform the prediction.

Unlike Kriging, linear interpolation has no fitted spatial-correlation model
(no variogram) — it's the cheaper, purely geometric end of the interpolation
spectrum, sitting between "no baseline at all" and `../interpolation_baseline`.

## No `train.sh`

Same as the Kriging baseline — no fitted/persisted state, only
`scripts/{local,slurm}/evaluate.sh`.

## Dependency

Only `scipy` and `joblib`, both listed in the project's `environment.yml`.

## Running

```bash
# Local sanity check (cpu, 20 samples, does not write results/):
./esdc/experiments/linear_interpolation_baseline/scripts/local/evaluate.sh

# HPC (full dataset, writes results/):
sbatch esdc/experiments/linear_interpolation_baseline/scripts/slurm/evaluate.sh
```
