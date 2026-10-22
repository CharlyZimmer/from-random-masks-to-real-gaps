"""Ordinary Kriging interpolation baseline on the held-out ESDC gap-filling dataset.

Non-learned baseline for `esdc/experiments/`: for each sample, fits an Ordinary
Kriging model (pykrige) on the observed (non-NaN) LST pixels of the middle
frame and predicts the synthetic gap pixels from spatial structure alone. No
model, no checkpoint, no auxiliary variable, no temporal context — this is the
naive-interpolation reference point for the learned spatiotemporal models.

Reuses ESDCGapFillingEvalDataset from evaluate_gapfilling.py (same eval
dataset, same gap-pixel definition, same MAE aggregation convention), so
metrics.json/run_info.json stay compatible with collect_results.py.

Run from project root:
    PYTHONPATH=. python esdc/src/evaluation/evaluate_kriging_baseline.py \
        --data_dir esdc/data/lst_gap_filling/dataset.h5 --num_samples 20
"""
import argparse
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from pykrige.ok import OrdinaryKriging

repo_root = os.environ.get(
    "GAP_AWARE_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", ".."),
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from esdc.src.evaluation.evaluate_gapfilling import ESDCGapFillingEvalDataset

_DEFAULT_H5 = os.path.join(
    os.environ.get("ESDC_GAP_FILLING_DIR", "./esdc/data/lst_gap_filling"),
    "dataset.h5",
)


# ---------------------------------------------------------------------------
# Per-sample Kriging
# ---------------------------------------------------------------------------

def _krige_sample(x, gt_middle, gap_mask, variogram_model: str):
    """Fit Ordinary Kriging on the observed LST pixels of the middle frame and
    predict the gap pixels. Returns (abs_err_sum, count, gap_fraction) or None
    if the sample can't be evaluated (no gap pixels or too few observed
    points to fit a variogram).
    """
    lst_mid = x[2, 0].numpy()          # (H, W) middle frame, LST channel
    gt = gt_middle[0].numpy()          # (H, W)
    mask = gap_mask.numpy()            # (H, W), 1 = gap pixel

    H, W = lst_mid.shape
    gap_fraction = float(mask.sum() / (H * W))

    obs_rows, obs_cols = np.where(~np.isnan(lst_mid))
    gap_rows, gap_cols = np.where((mask > 0) & ~np.isnan(gt))

    if len(gap_rows) == 0 or len(obs_rows) < 3:
        return None

    ok = OrdinaryKriging(
        obs_cols.astype(float), obs_rows.astype(float), lst_mid[obs_rows, obs_cols],
        variogram_model=variogram_model,
        verbose=False, enable_plotting=False,
    )
    pred, _ = ok.execute("points", gap_cols.astype(float), gap_rows.astype(float))
    pred = np.asarray(pred)

    target = gt[gap_rows, gap_cols]
    abs_err_sum = float(np.abs(pred - target).sum())
    count = len(gap_rows)

    return abs_err_sum, count, gap_fraction


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(dataset, variogram_model: str, n_jobs: int):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_krige_sample)(*dataset[i], variogram_model)
        for i in range(len(dataset))
    )
    results = [r for r in results if r is not None]

    if not results:
        print("WARNING: no valid gap pixels found across all samples.")
        return None

    abs_err_sum = sum(r[0] for r in results)
    count = sum(r[1] for r in results)
    gap_fractions = [r[2] for r in results]

    mean_mae = abs_err_sum / count

    return {
        "mean_mae": mean_mae,
        "per_band_mae": [mean_mae],
        "n_samples": len(results),
        "gap_coverage_mean": float(np.mean(gap_fractions)),
        "gap_coverage_std": float(np.std(gap_fractions)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ordinary Kriging ESDC gap-filling baseline")
    p.add_argument("--data_dir", default=_DEFAULT_H5,
                   help="Path to the gap-filling dataset.h5 file")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Max samples to evaluate (default: all)")
    p.add_argument("--variogram_model", default="linear",
                   help="pykrige variogram model (linear, power, gaussian, spherical, exponential)")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel workers for per-sample Kriging fits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default=None,
                   help="Directory to write metrics.json and run_info.json (for tracked experiment runs)")
    p.add_argument("--experiment_id", default=None,
                   help="Experiment identifier written into run_info.json")
    return p.parse_args()


def _load_norm_range(data_dir: str):
    """Read meta.json from data_dir and return the normalization range used."""
    meta_path = os.path.join(data_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get('valid_range') or meta.get('computed_range')


def _write_results(args, results, norm_range):
    """Write metrics.json and run_info.json to --results_dir when provided."""
    if args.results_dir is None:
        return

    import datetime
    import subprocess as _sp

    os.makedirs(args.results_dir, exist_ok=True)
    scale = (norm_range[1] - norm_range[0]) if norm_range is not None else None

    metrics = {
        "mean_mae_normalized": results["mean_mae"],
        "mean_mae_kelvin": round(results["mean_mae"] * scale, 4) if scale else None,
        "per_band_mae_normalized": results["per_band_mae"],
        "per_band_mae_kelvin": [round(v * scale, 4) for v in results["per_band_mae"]] if scale else None,
        "n_samples": results["n_samples"],
        "gap_coverage_mean": results["gap_coverage_mean"],
        "gap_coverage_std": results["gap_coverage_std"],
        "norm_range": list(norm_range) if norm_range else None,
    }
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to: {metrics_path}")

    try:
        git_commit = _sp.check_output(
            ["git", "rev-parse", "HEAD"], stderr=_sp.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = None

    exp_id = args.experiment_id or os.path.basename(
        os.path.dirname(os.path.abspath(args.results_dir))
    )

    run_info = {
        "experiment_id": exp_id,
        "eval_dataset_path": os.path.abspath(args.data_dir),
        "eval_date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "method": {
            "name": "ordinary_kriging",
            "library": "pykrige",
            "variogram_model": args.variogram_model,
            "n_jobs": args.n_jobs,
        },
    }
    run_info_path = os.path.join(args.results_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info written to: {run_info_path}")


def main():
    args = parse_args()

    dataset = ESDCGapFillingEvalDataset(args.data_dir)
    print(f"Dataset size: {len(dataset)} samples")

    if args.num_samples is not None and args.num_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), size=args.num_samples, replace=False).tolist()
        dataset = [dataset[i] for i in indices]
        print(f"Evaluating on {args.num_samples} randomly selected samples")

    print(f"Running Ordinary Kriging (variogram={args.variogram_model}, n_jobs={args.n_jobs})...")
    results = evaluate(dataset, args.variogram_model, args.n_jobs)

    if results is None:
        return

    norm_range = _load_norm_range(os.path.dirname(os.path.abspath(args.data_dir)))
    scale = (norm_range[1] - norm_range[0]) if norm_range is not None else None

    print("\n=== Kriging Baseline Evaluation Results ===")
    print(f"Samples evaluated : {results['n_samples']}")
    print(f"Gap coverage      : {results['gap_coverage_mean']*100:.1f}% ± {results['gap_coverage_std']*100:.1f}%")
    if scale is not None:
        print(f"Mean MAE (gap px) : {results['mean_mae']:.4f}  [{results['mean_mae']*scale:.2f} K]")
    else:
        print(f"Mean MAE (gap px) : {results['mean_mae']:.4f}")

    _write_results(args, results, norm_range)


if __name__ == "__main__":
    main()
