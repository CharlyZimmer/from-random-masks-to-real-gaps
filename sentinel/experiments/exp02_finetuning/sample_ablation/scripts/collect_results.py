"""Aggregate the sample-count ablation results into a Markdown table.

Walks sample_ablation/results/n*/, reads each variant's metrics.json/run_info.json
(written by scripts/slurm/evaluate_n*.sh), and writes sample_ablation/results/RESULTS.md.
Also pulls in exp04_scratch_gapfilling as a reference row, read directly from its own
tracked results/metrics.json.

Sentinel port of esdc/experiments/exp02_finetuning/sample_ablation/scripts/collect_results.py:
no MAE(K) column -- Sentinel has no single physical unit across its 10 bands, only
normalized MAE is reported (see evaluate_gapfilling.py::_write_results).

This is a separate script from sentinel/experiments/scripts/collect_results.py (which
aggregates the main experiments) and does not modify it or the main RESULTS.md.

Usage (from project root):
    python sentinel/experiments/exp02_finetuning/sample_ablation/scripts/collect_results.py
"""
import json
import os
import re
import sys

import yaml

_LABEL_RE = re.compile(r"^n(\d+)$")


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


def _variant_rows(sample_ablation_dir):
    configs_dir = os.path.join(sample_ablation_dir, "configs")
    results_dir = os.path.join(sample_ablation_dir, "results")

    rows = []
    for name in sorted(os.listdir(configs_dir)):
        if not name.endswith(".yaml"):
            continue
        label = name[: -len(".yaml")]
        m = _LABEL_RE.match(label)
        if not m:
            continue
        n_samples = int(m.group(1))

        config_path = os.path.join(configs_dir, name)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        exp_id = cfg["experiment"]["id"]

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

        rows.append((exp_id, n_samples, status, train_loss, val_loss, mae_norm, eval_n, eval_date))

    rows.sort(key=lambda r: r[1])
    return rows


def _reference_row(repo_root, exp_dir_name, exp_id, n_label):
    exp_dir = os.path.join(repo_root, "sentinel", "experiments", exp_dir_name)
    metrics = _read_json(os.path.join(exp_dir, "results", "metrics.json"))
    run_info = _read_json(os.path.join(exp_dir, "results", "run_info.json"))

    if metrics is None:
        print(f"Note: {exp_dir_name}/results/metrics.json not found, skipping reference row", file=sys.stderr)
        return None

    mae_norm = f"{metrics['mean_mae_normalized']:.4f}"
    eval_n = str(metrics["n_samples"])
    eval_date = run_info.get("eval_date", "—")[:10] if run_info is not None else "—"
    train_loss, val_loss = _loss_cells(metrics)

    return (f"{exp_id} (reference)", n_label, "done", train_loss, val_loss, mae_norm, eval_n, eval_date)


def main():
    repo_root = _repo_root()
    sample_ablation_dir = os.path.join(
        repo_root, "sentinel", "experiments", "exp02_finetuning", "sample_ablation"
    )
    results_path = os.path.join(sample_ablation_dir, "results", "RESULTS.md")

    rows = _variant_rows(sample_ablation_dir)

    ref_rows = []
    for exp_dir_name, exp_id, n_label in [
        ("exp04_scratch_gapfilling", "exp04_scratch_gapfilling", "n/a (from scratch)"),
    ]:
        r = _reference_row(repo_root, exp_dir_name, exp_id, n_label)
        if r is not None:
            ref_rows.append(r)

    header = (
        "# exp02_finetuning — Sample-Count Ablation Results\n\n"
        "Sub-experiment of `exp02_finetuning` (see `../README.md`). Same pretrained "
        "checkpoint (`exp01_zero_shot`), same effective peak LR (4e-5) and LR schedule "
        "shape, same model config (192/12/3, patch_size=4) as `exp02_finetuning` — only "
        "the number of fine-tuning samples drawn (seeded, seed=42) from "
        "`$S2_LMDB_PATH` joined with `$S2_GAP_FILLING_TRAINING_DIR/lmdb/gap_masks.lmdb` "
        "varies, and training runs for 20 epochs, the same as exp02. The 3 "
        "smallest variants (n0050/n0100/n0200) train on a single GPU instead of 4-way "
        "DDP (4-way DDP leaves too few steps/epoch per replica at this scale with "
        "batch_size=8) — `base_lr` is rescaled accordingly (`4e-5` vs `1e-5` for "
        "n0400-n12800) so the actual peak LR stays identical across all 9 variants; "
        "see the individual `configs/n*.yaml` files.\n\n"
        "\"N samples\" below is the number of *fine-tuning* samples used, not the "
        "eval-set size — every row is evaluated on the same canonical, fixed eval set "
        "(`$S2_DATA_DIR/gap_filling`, `N eval samples` column) used by every other "
        "experiment in this repo. MAE is computed over synthetic gap pixels in the "
        "middle frame, averaged across all 10 bands. Sentinel has no single physical "
        "unit like ESDC's Kelvin, so only normalized ([0,1] reflectance) MAE is shown.\n"
    )
    lines = [
        header,
        "| ID | N samples | Status | Train loss | Val loss | MAE (norm) | N eval samples | Eval date |",
        "|----|-----------|--------|-----------|---------|-----------|-----------------|-----------|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
    for r in ref_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
    lines.append("")
    lines.append("![Eval loss vs number of fine-tuning samples](loss_vs_samples.png)")
    lines.append("")
    lines.append(
        "_Regenerate with: `python sentinel/experiments/exp02_finetuning/sample_ablation/scripts/collect_results.py` "
        "(writes this file directly). Regenerate the plot with "
        "`python sentinel/experiments/exp02_finetuning/sample_ablation/scripts/plot_results.py`._"
    )
    content = "\n".join(lines) + "\n"

    with open(results_path, "w") as f:
        f.write(content)
    print(content, end="")
    print(f"Wrote {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
