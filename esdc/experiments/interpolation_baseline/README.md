# interpolation_baseline

Non-learned baseline: Ordinary Kriging spatial interpolation, evaluated on the
same canonical gap-filling dataset as every other experiment
(`$ESDC_GAP_FILLING_DIR/dataset.h5`, see `../eval_dataset.yaml`).

## Method

For each sample, fits an Ordinary Kriging model (`pykrige.ok.OrdinaryKriging`)
on the observed (non-NaN) LST pixels of the middle frame, and predicts the
synthetic gap pixels from spatial structure alone.

Intentionally out of scope, to keep this a naive baseline:
- **Auxiliary variable** (`air_temperature_2m`) — fully observed at gap pixels
  in the middle frame (unlike LST, only channel 0 is gap-masked — see
  `patchfill/dataset_building.py::generate_synthetic_gaps`), but not used here.
  A stronger baseline could use it as an external drift (Universal Kriging).
- **Temporal context** — the other 4 frames of the 5-frame input series are
  ignored; only the middle frame's spatial neighbors inform the prediction.

This makes the comparison against `../exp04_scratch_gapfilling` (labeled
"baseline" in `../RESULTS.md`) and the zero-shot experiments (`exp01`, `exp03`)
a fair "classical interpolation vs. learned spatiotemporal model" contrast.

## No `train.sh`

Kriging has no fitted/persisted state — the variogram is fit fresh per sample
at evaluation time. Only `scripts/{local,slurm}/evaluate.sh` exist.

## Dependency

Requires `pykrige` and `joblib`, both listed in the project's
`environment.yml`.

## Running

```bash
# Local sanity check (cpu, 20 samples, does not write results/):
./esdc/experiments/interpolation_baseline/scripts/local/evaluate.sh

# HPC (full dataset, writes results/):
sbatch esdc/experiments/interpolation_baseline/scripts/slurm/evaluate.sh
```
