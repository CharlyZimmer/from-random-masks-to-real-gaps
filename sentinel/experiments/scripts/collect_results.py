"""Aggregate experiment results into a Markdown comparison table.

Walks sentinel/experiments/, reads each experiment's config.yaml and
results/metrics.json, writes the table to sentinel/experiments/RESULTS.md,
and also prints it to stdout.

Sentinel port of esdc/experiments/scripts/collect_results.py: no ablations
subtree (Sentinel's ablations have their own collect_results.py scripts), and
no Kelvin column (Sentinel has no single physical unit across its 10
bands — only normalized MAE is reported).

Usage (from project root):
    python sentinel/experiments/scripts/collect_results.py
"""
import json
import os
import sys

import yaml


def _find_experiment_dirs(experiments_root):
    dirs = []
    for name in sorted(os.listdir(experiments_root)):
        path = os.path.join(experiments_root, name)
        if os.path.isdir(path) and (
            name.startswith("exp")
            or name in ("interpolation_baseline", "linear_interpolation_baseline")
        ):
            dirs.append(path)
    return dirs


def main():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    experiments_root = os.path.join(repo_root, "sentinel", "experiments")
    results_path = os.path.join(experiments_root, "RESULTS.md")

    rows = []
    for exp_dir in _find_experiment_dirs(experiments_root):
        config_path  = os.path.join(exp_dir, "config.yaml")
        metrics_path = os.path.join(exp_dir, "results", "metrics.json")
        run_info_path = os.path.join(exp_dir, "results", "run_info.json")

        if not os.path.exists(config_path):
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        exp_id = cfg["experiment"]["id"]
        desc   = cfg["experiment"]["description"].strip().rstrip(".")
        status = "done" if os.path.exists(metrics_path) else "pending"

        train_n = train_loss = val_loss = "—"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            mae_norm = f"{m['mean_mae_normalized']:.4f}"
            n        = str(m["n_samples"])
            # Filled in by evaluate_gapfilling.py (read from the checkpoint +
            # its TensorBoard event files); num_training_samples is added by
            # hand. All optional — a "—" just means not yet refreshed / N/A
            # (e.g. interpolation baselines have no training run).
            if m.get("num_training_samples") is not None:
                train_n = str(m["num_training_samples"])
            if m.get("train_loss_normalized") is not None:
                train_loss = f"{m['train_loss_normalized']:.4f}"
            if m.get("val_loss_normalized") is not None:
                val_loss = f"{m['val_loss_normalized']:.4f}"
        else:
            mae_norm = n = "—"

        eval_date = "—"
        if os.path.exists(run_info_path):
            with open(run_info_path) as f:
                ri = json.load(f)
            eval_date = ri.get("eval_date", "—")[:10]  # date only

        rows.append((exp_id, desc, status, train_n, train_loss, val_loss, mae_norm, n, eval_date))

    header = (
        "# Sentinel-2 Gap-Filling Results\n\n"
        "Canonical test set: `$S2_DATA_DIR/gap_filling`  \n"
        "(HPC: full disjoint-location eval set; local: small subset for pipeline sanity checks only)\n\n"
        "MAE is computed over synthetic gap pixels in the middle frame, averaged across all 10 bands.\n"
        "Values are normalized ([0,1] reflectance) MAE — Sentinel has no single physical unit like ESDC's Kelvin.  \n"
        "Train / Val loss are the best-epoch training and validation MAE (norm) "
        "of the checkpoint used, read from the Lightning logs. "
        "\"Train N\" (training-set size) is entered manually. "
        "All three are \"—\" for the interpolation baselines (no training run).\n"
    )
    lines = [
        header,
        "| ID | Method | Status | Train N | Train loss | Val loss | MAE (norm) | N samples | Eval date |",
        "|----|--------|--------|---------|-----------|---------|-----------|-----------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} |"
        )
    lines.append("")
    lines.append(
        "_Regenerate with: `python sentinel/experiments/scripts/collect_results.py` "
        "(writes this file directly; stdout redirection is not required)_"
    )
    content = "\n".join(lines) + "\n"

    with open(results_path, "w") as f:
        f.write(content)
    print(content, end="")
    print(f"Wrote {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
