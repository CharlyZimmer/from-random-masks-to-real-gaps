"""Linear interpolation baseline on the held-out Sentinel-2 gap-filling dataset.

Non-learned baseline for `sentinel/experiments/`: for each sample and each of
the 10 bands independently, interpolates the synthetic gap pixels of the
middle frame from the observed (non-NaN) pixels via piecewise-linear
interpolation over their Delaunay triangulation (scipy.interpolate.griddata).
No model, no checkpoint, no temporal context, no fitted spatial-correlation
structure (unlike the Kriging baseline) — the cheapest possible interpolation
reference point.

Reuses GapFillingEvalDataset from evaluate_gapfilling.py (same eval dataset,
same gap-pixel definition, same MAE aggregation convention), so
metrics.json/run_info.json stay compatible with collect_results.py.

Run from project root:
    PYTHONPATH=. python sentinel/src/evaluation/evaluate_linear_interpolation_baseline.py \
        --data_dir sentinel/data/gap_filling --num_samples 5
"""
import argparse
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import griddata
from scipy.spatial import QhullError

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
# Per-sample, per-band linear interpolation
# ---------------------------------------------------------------------------

def _linear_interp_band(mid_band, gt_band, gap_mask):
    """Interpolate one band's observed pixels at the gap pixels via linear
    interpolation over their Delaunay triangulation. Points outside the
    convex hull of the observed pixels (which griddata's "linear" method
    leaves as NaN) are filled in with nearest-neighbor interpolation. If the
    observed pixels are degenerate for triangulation, Qhull raises
    QhullError; fall back to nearest-neighbor for the whole band in that case.
    Returns (abs_err_sum, count) or None if the band can't be evaluated.
    """
    obs_rows, obs_cols = np.where(~np.isnan(mid_band))
    gap_rows, gap_cols = np.where((gap_mask > 0) & ~np.isnan(gt_band))

    if len(gap_rows) == 0 or len(obs_rows) < 3:
        return None

    obs_points = np.column_stack((obs_cols, obs_rows)).astype(float)
    obs_values = mid_band[obs_rows, obs_cols]
    gap_points = np.column_stack((gap_cols, gap_rows)).astype(float)

    try:
        pred = griddata(obs_points, obs_values, gap_points, method="linear")
    except QhullError:
        pred = np.full(len(gap_points), np.nan)

    nan_mask = np.isnan(pred)
    if nan_mask.any():
        pred[nan_mask] = griddata(
            obs_points, obs_values, gap_points[nan_mask], method="nearest"
        )

    target = gt_band[gap_rows, gap_cols]
    abs_err_sum = float(np.abs(pred - target).sum())
    count = len(gap_rows)

    return abs_err_sum, count


def _linear_interp_sample(x, gt_middle, gap_mask):
    """Interpolate independently per band. Returns
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

    per_band_sum = np.zeros(C)
    per_band_count = np.zeros(C)
    any_valid = False

    for c in range(C):
        result = _linear_interp_band(mid[c], gt[c], mask)
        if result is not None:
            per_band_sum[c], per_band_count[c] = result
            any_valid = True

    if not any_valid:
        return None
    return per_band_sum, per_band_count, gap_fraction


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(dataset, n_jobs: int):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_linear_interp_sample)(*dataset[i])
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
    p = argparse.ArgumentParser(description="Linear interpolation Sentinel-2 gap-filling baseline")
    p.add_argument("--data_dir", default=_DEFAULT_DATA_DIR,
                   help="Path to gap_filling/ directory")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Max samples to evaluate (default: all)")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel workers for per-sample interpolation")
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
            "name": "linear_interpolation",
            "library": "scipy.interpolate.griddata",
            "fallback": "nearest (points outside convex hull, or whole band if triangulation is degenerate)",
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

    print(f"Running linear interpolation (n_jobs={args.n_jobs})...")
    results = evaluate(dataset, args.n_jobs)

    if results is None:
        return

    print("\n=== Linear Interpolation Baseline Evaluation Results ===")
    print(f"Samples evaluated : {results['n_samples']}")
    print(f"Gap coverage      : {results['gap_coverage_mean']*100:.1f}% ± {results['gap_coverage_std']*100:.1f}%")
    print(f"Mean MAE (gap px) : {results['mean_mae']:.4f}")
    print("Per-band MAE:")
    for i, mae in enumerate(results["per_band_mae"]):
        print(f"  Band {i}: {mae:.4f}")

    _write_results(args, results)


if __name__ == "__main__":
    main()
