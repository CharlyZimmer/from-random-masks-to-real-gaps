"""Ordinary Kriging interpolation baseline on the held-out Sentinel-2 gap-filling dataset.

Non-learned baseline for `sentinel/experiments/`: for each sample and each of
the 10 bands independently, fits an Ordinary Kriging model (pykrige) on the
observed (non-NaN) pixels of the middle frame and predicts the synthetic gap
pixels from spatial structure alone. No model, no checkpoint, no temporal
context — the naive-interpolation reference point for the learned
spatiotemporal models.

At 128x128 (16384 px) x 10 bands, fitting Kriging on every observed pixel is
not tractable (pykrige's variogram/kriging-matrix solve is O(n^2)-O(n^3) in
the number of observed points). Unlike ESDC's 64x64x1 baseline, this script
therefore randomly subsamples the observed pixels of each band down to
--max_obs_points (default 2000) before fitting — a deliberate tractability
tradeoff, not present in the ESDC baseline.

Reuses GapFillingEvalDataset from evaluate_gapfilling.py (same eval dataset,
same gap-pixel definition, same MAE aggregation convention), so
metrics.json/run_info.json stay compatible with collect_results.py.

Run from project root:
    PYTHONPATH=. python sentinel/src/evaluation/evaluate_kriging_baseline.py \
        --data_dir sentinel/data/gap_filling --num_samples 5
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

from sentinel.src.evaluation.evaluate_gapfilling import GapFillingEvalDataset

_DEFAULT_DATA_DIR = os.path.join(
    os.environ.get("S2_DATA_DIR", "./sentinel/data"),
    "gap_filling",
)


# ---------------------------------------------------------------------------
# Per-sample, per-band Kriging
# ---------------------------------------------------------------------------

def _krige_band(mid_band, gt_band, gap_mask, variogram_model: str, max_obs_points: int, rng):
    """Fit Ordinary Kriging on one band's observed pixels and predict the gap
    pixels. Returns (abs_err_sum, count) or None if the band can't be
    evaluated (no gap pixels or too few observed points to fit a variogram).
    """
    obs_rows, obs_cols = np.where(~np.isnan(mid_band))
    gap_rows, gap_cols = np.where((gap_mask > 0) & ~np.isnan(gt_band))

    if len(gap_rows) == 0 or len(obs_rows) < 3:
        return None

    if len(obs_rows) > max_obs_points:
        keep = rng.choice(len(obs_rows), size=max_obs_points, replace=False)
        obs_rows, obs_cols = obs_rows[keep], obs_cols[keep]

    obs_values = mid_band[obs_rows, obs_cols]
    target = gt_band[gap_rows, gap_cols]

    if np.ptp(obs_values) == 0:
        # Zero-variance neighborhood (e.g. all-nodata-padding pixels left
        # observed after masking): semivariance is flat everywhere, so
        # pykrige's variogram-parameter least-squares fit degenerates (its
        # sill bound collapses to a single point and raises ValueError).
        # The correct Ordinary Kriging prediction here is just the constant
        # observed value, so short-circuit the fit instead of crashing.
        pred = np.full(len(gap_rows), obs_values[0])
    else:
        ok = OrdinaryKriging(
            obs_cols.astype(float), obs_rows.astype(float), obs_values,
            variogram_model=variogram_model,
            verbose=False, enable_plotting=False,
        )
        pred, _ = ok.execute("points", gap_cols.astype(float), gap_rows.astype(float))
        pred = np.asarray(pred)
    abs_err_sum = float(np.abs(pred - target).sum())
    count = len(gap_rows)

    return abs_err_sum, count


def _krige_sample(x, gt_middle, gap_mask, variogram_model: str, max_obs_points: int, seed: int):
    """Fit Kriging independently per band. Returns
    (per_band_abs_err_sum, per_band_count, gap_fraction) or None if no band
    could be evaluated.
    """
    # GapFillingEvalDataset returns plain numpy arrays (unlike ESDC's
    # torch-tensor dataset), so no .numpy() conversion is needed here.
    mid = np.asarray(x[2])          # (C, H, W) middle frame
    gt = np.asarray(gt_middle)      # (C, H, W)
    mask = np.asarray(gap_mask)     # (H, W), 1 = gap pixel

    C, H, W = mid.shape
    gap_fraction = float(mask.sum() / (H * W))

    rng = np.random.default_rng(seed)
    per_band_sum = np.zeros(C)
    per_band_count = np.zeros(C)
    any_valid = False

    for c in range(C):
        result = _krige_band(mid[c], gt[c], mask, variogram_model, max_obs_points, rng)
        if result is not None:
            per_band_sum[c], per_band_count[c] = result
            any_valid = True

    if not any_valid:
        return None
    return per_band_sum, per_band_count, gap_fraction


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(dataset, variogram_model: str, max_obs_points: int, n_jobs: int, seed: int):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_krige_sample)(*dataset[i], variogram_model, max_obs_points, seed + i)
        for i in range(len(dataset))
    )
    results = [r for r in results if r is not None]

    if not results:
        print("WARNING: no valid gap pixels found across all samples.")
        return None

    C = results[0][0].shape[0]
    per_band_sum = np.zeros(C)
    per_band_count = np.zeros(C)
    gap_fractions = []
    for band_sum, band_count, gap_fraction in results:
        per_band_sum += band_sum
        per_band_count += band_count
        gap_fractions.append(gap_fraction)

    per_band_mae = per_band_sum / np.clip(per_band_count, 1, None)
    mean_mae = float(per_band_mae[per_band_count > 0].mean())

    return {
        "mean_mae": mean_mae,
        "per_band_mae": per_band_mae.tolist(),
        "n_samples": len(results),
        "gap_coverage_mean": float(np.mean(gap_fractions)),
        "gap_coverage_std": float(np.std(gap_fractions)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ordinary Kriging Sentinel-2 gap-filling baseline")
    p.add_argument("--data_dir", default=_DEFAULT_DATA_DIR,
                   help="Path to gap_filling/ directory")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Max samples to evaluate (default: all)")
    p.add_argument("--variogram_model", default="linear",
                   help="pykrige variogram model (linear, power, gaussian, spherical, exponential)")
    p.add_argument("--max_obs_points", type=int, default=2000,
                   help="Cap on observed pixels used per band's Kriging fit (subsampled if exceeded)")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel workers for per-sample Kriging fits")
    p.add_argument("--normalize", action="store_true",
                   help="Apply Sentinel-2 normalization (÷10000, clip [0,1]) — must match training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default=None,
                   help="Directory to write metrics.json and run_info.json (for tracked experiment runs)")
    p.add_argument("--experiment_id", default=None,
                   help="Experiment identifier written into run_info.json")
    return p.parse_args()


def _write_results(args, results):
    """Write metrics.json and run_info.json to --results_dir when provided."""
    if args.results_dir is None:
        return

    import datetime
    import subprocess as _sp

    os.makedirs(args.results_dir, exist_ok=True)

    metrics = {
        "mean_mae_normalized": results["mean_mae"],
        "per_band_mae_normalized": results["per_band_mae"],
        "n_samples": results["n_samples"],
        "gap_coverage_mean": results["gap_coverage_mean"],
        "gap_coverage_std": results["gap_coverage_std"],
        "normalized": bool(args.normalize),
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
            "max_obs_points": args.max_obs_points,
            "n_jobs": args.n_jobs,
        },
    }
    run_info_path = os.path.join(args.results_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info written to: {run_info_path}")


def main():
    args = parse_args()

    dataset = GapFillingEvalDataset(args.data_dir, normalize=args.normalize)
    print(f"Dataset size: {len(dataset)} samples")

    if args.num_samples is not None and args.num_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), size=args.num_samples, replace=False).tolist()
        dataset = [dataset[i] for i in indices]
        print(f"Evaluating on {args.num_samples} randomly selected samples")

    print(f"Running Ordinary Kriging (variogram={args.variogram_model}, "
          f"max_obs_points={args.max_obs_points}, n_jobs={args.n_jobs})...")
    results = evaluate(dataset, args.variogram_model, args.max_obs_points, args.n_jobs, args.seed)

    if results is None:
        return

    print("\n=== Kriging Baseline Evaluation Results ===")
    print(f"Samples evaluated : {results['n_samples']}")
    print(f"Gap coverage      : {results['gap_coverage_mean']*100:.1f}% ± {results['gap_coverage_std']*100:.1f}%")
    print(f"Mean MAE (gap px) : {results['mean_mae']:.4f}")
    print("Per-band MAE:")
    for i, mae in enumerate(results["per_band_mae"]):
        print(f"  Band {i}: {mae:.4f}")

    _write_results(args, results)


if __name__ == "__main__":
    main()
