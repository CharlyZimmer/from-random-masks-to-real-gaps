"""Aggregate the max-gap-ratio ablation results into a Markdown table.

Walks max_gap_ablation/configs/*.yaml, reads each variant's data.max_gap_ratio
plus results/gapXXX/{metrics.json,run_info.json} (written by
scripts/slurm/evaluate_gap*.sh), and writes max_gap_ablation/results/RESULTS.md.
Also pulls in exp01_zero_shot (the pre-training LMDB's own creation-time cap,
max_gap_ratio=0.2, effectively unfiltered relative to this ablation's post-hoc
filter) as a reference row, read directly from its own tracked results/metrics.json.

Sentinel port of esdc/experiments/exp01_zero_shot/max_gap_ablation/scripts/collect_results.py:
single table only -- sentinel/src/evaluation/evaluate_gapfilling.py has no
gap_ratio_groups natural-gap breakdown (unlike ESDC's), and Sentinel has no
single physical unit across its 10 bands, so only normalized MAE is reported
(see evaluate_gapfilling.py::_write_results). Same structure as
sentinel/experiments/exp02_finetuning/sample_ablation/scripts/collect_results.py.

This is a separate script from sentinel/experiments/scripts/collect_results.py (which
aggregates the main experiments) and does not modify it or the main RESULTS.md.

Usage (from project root):
    python sentinel/experiments/exp01_zero_shot/max_gap_ablation/scripts/collect_results.py
"""
import json
import os
import sys

import yaml


def _repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _loss_cells(metrics):
    """(train_loss, val_loss) normalized-MAE cells, "—" until (re-)evaluated.

    Written into metrics.json by evaluate_gapfilling.py from the checkpoint +
    its TensorBoard event files.
    """
    if not metrics:
        return "—", "—"
    tl = metrics.get("train_loss_normalized")
    vl = metrics.get("val_loss_normalized")
    return (
        f"{tl:.4f}" if tl is not None else "—",
        f"{vl:.4f}" if vl is not None else "—",
    )


def _variant_rows(ablation_dir):
    configs_dir = os.path.join(ablation_dir, "configs")
    results_dir = os.path.join(ablation_dir, "results")

    rows = []
    for name in sorted(os.listdir(configs_dir)):
        if not name.endswith(".yaml"):
            continue
        label = name[: -len(".yaml")]

        config_path = os.path.join(configs_dir, name)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        exp_id = cfg["experiment"]["id"]
        max_gap_ratio = cfg["data"]["max_gap_ratio"]

        metrics = _read_json(os.path.join(results_dir, label, "metrics.json"))
        run_info = _read_json(os.path.join(results_dir, label, "run_info.json"))

        status = "done" if metrics is not None else "pending"
        if metrics is not None:
            mae_norm = f"{metrics['mean_mae_normalized']:.4f}"
            eval_n = str(metrics["n_samples"])
        else:
            mae_norm = eval_n = "—"

        train_loss, val_loss = _loss_cells(metrics)

        eval_date = "—"
        if run_info is not None:
            eval_date = run_info.get("eval_date", "—")[:10]

        rows.append((exp_id, max_gap_ratio, status, train_loss, val_loss,
                     mae_norm, eval_n, eval_date))

    rows.sort(key=lambda r: r[1])
    return rows


def _reference_row(repo_root):
    exp_dir = os.path.join(repo_root, "sentinel", "experiments", "exp01_zero_shot")
    metrics = _read_json(os.path.join(exp_dir, "results", "metrics.json"))
    run_info = _read_json(os.path.join(exp_dir, "results", "run_info.json"))

    if metrics is None:
        print("Note: exp01_zero_shot/results/metrics.json not found, skipping reference row", file=sys.stderr)
        return None

    mae_norm = f"{metrics['mean_mae_normalized']:.4f}"
    eval_n = str(metrics["n_samples"])
    eval_date = run_info.get("eval_date", "—")[:10] if run_info is not None else "—"
    train_loss, val_loss = _loss_cells(metrics)

    return ("exp01_zero_shot (reference, unfiltered)", 0.2, "done", train_loss,
            val_loss, mae_norm, eval_n, eval_date)


def main():
    repo_root = _repo_root()
    ablation_dir = os.path.join(
        repo_root, "sentinel", "experiments", "exp01_zero_shot", "max_gap_ablation"
    )
    results_path = os.path.join(ablation_dir, "results", "RESULTS.md")

    rows = _variant_rows(ablation_dir)
    ref_row = _reference_row(repo_root)
    if ref_row is not None:
        rows.append(ref_row)

    header = (
        "# exp01_zero_shot — Max-Gap-Ratio Ablation Results (Sentinel)\n\n"
        "Sub-experiment of `exp01_zero_shot` (see `../README.md` and this "
        "directory's own `README.md`). Same architecture, hyperparameters, and "
        "eval dataset as `exp01_zero_shot` — only the pre-training LMDB "
        "(`$S2_LMDB_PATH`) is post-hoc filtered (at training time, not via a "
        "separate dataset) to keep samples whose middle-frame band-0 NaN ratio "
        "is <= max_gap_ratio, vs. the pre-training LMDB's own 0.2 creation-time "
        "cap (`exp01_zero_shot`, effectively unfiltered relative to this "
        "ablation). MAE is computed over synthetic gap pixels in the middle "
        "frame, averaged across all 10 bands, on the same canonical eval set "
        "(`$S2_DATA_DIR/gap_filling`) used by every other experiment in this "
        "repo. Sentinel has no single physical unit like ESDC's Kelvin, so "
        "only normalized ([0,1] reflectance) MAE is shown, and (unlike ESDC's "
        "version of this ablation) there is no natural-gap-ratio subset "
        "breakdown -- sentinel's evaluate_gapfilling.py doesn't compute one.\n"
    )
    lines = [
        header,
        "| ID | max_gap_ratio | Status | Train loss | Val loss | MAE (norm) | N eval samples | Eval date |",
        "|----|----------------|--------|-----------|---------|-----------|-----------------|-----------|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
    lines.append("")
    lines.append(
        "_Regenerate with: `python sentinel/experiments/exp01_zero_shot/max_gap_ablation/scripts/collect_results.py` "
        "(writes this file directly)._"
    )
    content = "\n".join(lines) + "\n"

    with open(results_path, "w") as f:
        f.write(content)
    print(content, end="")
    print(f"Wrote {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
