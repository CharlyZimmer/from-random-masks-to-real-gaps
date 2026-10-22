"""Aggregate the max-gap-ratio ablation results into three Markdown tables.

Walks max_gap_ablation/configs/*.yaml, reads each variant's data.max_gap_ratio
plus results/<label>/metrics.json + run_info.json (written by
scripts/slurm/evaluate_gap*.sh), and writes max_gap_ablation/results/RESULTS.md.
Also pulls in exp01_zero_shot (the unfiltered, max_gap_ratio=0.2 baseline) as a
reference row, read directly from its own tracked results/metrics.json.

Produces three tables, one per gap_ratio_groups key written by
evaluate_gapfilling.py -- "full" (all eval samples), "max_0.10" (eval samples
whose target has <=10% natural, pre-donor-injection gap fraction), and
"max_0.00" (0% natural gap fraction, i.e. a fully clean ground truth patch).
This groups eval samples by how much of their target was naturally missing
before synthetic gap injection (bounded by max_gap_portion=0.2 at eval-set
creation time) -- independent of the donor-mask-driven gap_mask fraction the
loss itself is computed over (which is unrelated and much larger on HPC,
governed by gap_donor_min_ratio/gap_donor_max_ratio). metrics.json files
written before this grouping was added have no "gap_ratio_groups" key; their
max_0.10/max_0.00 rows show "—" until re-evaluated.

This is a separate script from esdc/experiments/scripts/collect_results.py (which
aggregates the main experiments + ablations/) and does not modify it or the main
RESULTS.md.

Usage (from project root):
    python esdc/experiments/exp01_zero_shot/max_gap_ablation/scripts/collect_results.py
"""
import json
import os
import sys

import yaml

GROUPS = [
    ("full", "Full evaluation dataset"),
    ("max_0.10", "Evaluation subset: ≤10% natural gap ratio in target"),
    ("max_0.00", "Evaluation subset: 0% natural gap ratio in target (fully clean ground truth)"),
]


def _repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _group_row(exp_id, max_gap_ratio, metrics, run_info, group_key):
    group = (metrics or {}).get("gap_ratio_groups", {}).get(group_key)

    # metrics.json written before grouping was added has no "gap_ratio_groups",
    # but its top-level keys already *are* the "full" group's numbers.
    if group is None and group_key == "full" and metrics is not None:
        group = {
            "mean_mae_normalized": metrics.get("mean_mae_normalized"),
            "mean_mae_kelvin": metrics.get("mean_mae_kelvin"),
            "n_samples": metrics.get("n_samples"),
        }

    if group is None:
        status = "pending" if metrics is None else "not re-evaluated"
        mae_norm = mae_k = eval_n = "—"
    elif group.get("mean_mae_normalized") is None:
        status = "no eligible samples"
        mae_norm = mae_k = "—"
        eval_n = str(group.get("n_samples", 0))
    else:
        status = "done"
        mae_norm = f"{group['mean_mae_normalized']:.4f}"
        mae_k = f"{group['mean_mae_kelvin']:.2f}" if group.get("mean_mae_kelvin") else "—"
        eval_n = str(group["n_samples"])

    eval_date = "—"
    if run_info is not None:
        eval_date = run_info.get("eval_date", "—")[:10]

    # Best-epoch train/val loss of the checkpoint (read from the Lightning logs
    # by evaluate_gapfilling.py). Run-level, not per eval-group, so repeated
    # across the three tables. "—" until the variant is (re-)evaluated.
    train_loss = (
        f"{metrics['train_loss_normalized']:.4f}"
        if metrics and metrics.get("train_loss_normalized") is not None else "—"
    )
    val_loss = (
        f"{metrics['val_loss_normalized']:.4f}"
        if metrics and metrics.get("val_loss_normalized") is not None else "—"
    )

    return (exp_id, max_gap_ratio, status, train_loss, val_loss,
            mae_norm, mae_k, eval_n, eval_date)


def _variant_rows(ablation_dir, group_key):
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

        rows.append(_group_row(exp_id, max_gap_ratio, metrics, run_info, group_key))

    rows.sort(key=lambda r: r[1])
    return rows


def _reference_row(repo_root, group_key):
    exp_dir = os.path.join(repo_root, "esdc", "experiments", "exp01_zero_shot")
    metrics = _read_json(os.path.join(exp_dir, "results", "metrics.json"))
    run_info = _read_json(os.path.join(exp_dir, "results", "run_info.json"))

    if metrics is None:
        print("Note: exp01_zero_shot/results/metrics.json not found, skipping reference row", file=sys.stderr)
        return None

    return _group_row(
        "exp01_zero_shot (reference, unfiltered)", 0.2, metrics, run_info, group_key
    )


def _render_table(title, rows):
    lines = [
        f"## {title}\n",
        "| ID | max_gap_ratio | Status | Train loss | Val loss | MAE (norm) | MAE (K) | N eval samples | Eval date |",
        "|----|----------------|--------|-----------|---------|-----------|---------|-----------------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} |"
        )
    return lines


def main():
    repo_root = _repo_root()
    ablation_dir = os.path.join(
        repo_root, "esdc", "experiments", "exp01_zero_shot", "max_gap_ablation"
    )
    results_path = os.path.join(ablation_dir, "results", "RESULTS.md")

    header = (
        "# exp01_zero_shot — Max-Gap-Ratio Ablation Results\n\n"
        "Sub-experiment of `exp01_zero_shot` (see `../README.md` and this "
        "directory's own `README.md`). Same architecture, hyperparameters, and "
        "eval dataset as `exp01_zero_shot` — only the pre-training LMDB is "
        "post-hoc filtered (at training time, not via a separate dataset) to "
        "keep samples whose middle-frame NaN ratio is <= max_gap_ratio, vs. the "
        "unfiltered (max_gap_ratio=0.2 at dataset-creation time) baseline. MAE "
        "is computed over synthetic gap pixels in the middle frame only, on the "
        "same canonical eval set (`$ESDC_GAP_FILLING_DIR/dataset.h5`) used by "
        "every other experiment in this repo.\n\n"
        "The three tables below additionally break the eval loss down by each "
        "evaluation sample's own *natural, pre-donor-injection* gap ratio in "
        "the target (bounded <=20% by `max_gap_portion` at eval-set creation "
        "time) — independent of the donor-mask-driven gap_mask fraction the "
        "loss is computed over. This tests whether the unfiltered model's edge "
        "is genuinely from training on gappy data, or just from matching the "
        "eval set's own gap-ratio composition: if the gap persists even on the "
        "low/zero-natural-gap-ratio subsets, that points to a real effect "
        "rather than distribution matching.\n"
    )

    lines = [header]
    for group_key, title in GROUPS:
        rows = _variant_rows(ablation_dir, group_key)
        ref_row = _reference_row(repo_root, group_key)
        if ref_row is not None:
            rows.append(ref_row)
        lines.extend(_render_table(title, rows))
        lines.append("")

    lines.append(
        "_Regenerate with: `python esdc/experiments/exp01_zero_shot/max_gap_ablation/scripts/collect_results.py` "
        "(writes this file directly)._"
    )
    content = "\n".join(lines) + "\n"

    with open(results_path, "w") as f:
        f.write(content)
    print(content, end="")
    print(f"Wrote {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
