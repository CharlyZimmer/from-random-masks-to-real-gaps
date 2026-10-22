"""Aggregate the model-size ablation results into a Markdown table.

Walks model_size_ablation/configs/*.yaml, reads each variant's
model.embed_dim/depth/num_heads plus results/<label>/metrics.json +
run_info.json (written by scripts/slurm/evaluate_all.sh /
evaluate_<label>.sh), and writes model_size_ablation/results/RESULTS.md.
Also pulls in exp01_zero_shot (the 192/12/3 baseline -- identical
architecture to the largest config tested here, not retrained) as a
reference row, read directly from its own tracked results/metrics.json +
results/run_info.json.

Produces a single table of full-dataset MAE. Unlike max_gap_ablation, this
ablation studies MAE vs. model capacity, not vs. gap-ratio composition, so it
does not need evaluate_gapfilling.py's natural-gap-ratio breakdown
("max_0.10"/"max_0.00") -- results/<label>/metrics.json is written with
--full_only, so only the "full" group is available (or, for metrics.json
files evaluated before --full_only existed, the "full" entry inside
gap_ratio_groups is used and the other groups are simply ignored).

This is a separate script from esdc/experiments/scripts/collect_results.py (which
aggregates the main experiments + ablations/) and does not modify it or the main
RESULTS.md.

Usage (from project root):
    python esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/collect_results.py
"""
import json
import os
import sys

import yaml

GROUPS = [
    ("full", "Full evaluation dataset"),
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


def _num_params(run_info):
    if run_info is None:
        return None
    return (run_info.get("model") or {}).get("num_params")


def _group_row(exp_id, arch, num_params, metrics, run_info, group_key):
    group = (metrics or {}).get("gap_ratio_groups", {}).get(group_key)

    if group is None and group_key == "full" and metrics is not None:
        group = {
            "mean_mae_normalized": metrics.get("mean_mae_normalized"),
            "mean_mae_kelvin": metrics.get("mean_mae_kelvin"),
            "n_samples": metrics.get("n_samples"),
        }

    if group is None:
        status = "pending" if metrics is None else "not re-evaluated"
        mae_norm = mae_k = "—"
        eval_n = "—"
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

    params_str = f"{num_params:,}" if num_params is not None else "—"

    # Best-epoch train/val loss of the checkpoint (read from the Lightning logs
    # by evaluate_gapfilling.py). "—" until the variant is (re-)evaluated.
    train_loss = (
        f"{metrics['train_loss_normalized']:.4f}"
        if metrics and metrics.get("train_loss_normalized") is not None else "—"
    )
    val_loss = (
        f"{metrics['val_loss_normalized']:.4f}"
        if metrics and metrics.get("val_loss_normalized") is not None else "—"
    )

    return (exp_id, arch, params_str, status, train_loss, val_loss,
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
        model_cfg = cfg["model"]
        arch = f"{model_cfg['embed_dim']}/{model_cfg['depth']}/{model_cfg['num_heads']}"

        metrics = _read_json(os.path.join(results_dir, label, "metrics.json"))
        run_info = _read_json(os.path.join(results_dir, label, "run_info.json"))
        num_params = _num_params(run_info)

        rows.append(_group_row(exp_id, arch, num_params, metrics, run_info, group_key))

    # Sort largest-to-smallest by (embed_dim, depth, num_heads), matching the
    # order the configs were designed in.
    rows.sort(key=lambda r: tuple(-int(x) for x in r[1].split("/")))
    return rows


def _reference_row(repo_root, group_key):
    exp_dir = os.path.join(repo_root, "esdc", "experiments", "exp01_zero_shot")
    metrics = _read_json(os.path.join(exp_dir, "results", "metrics.json"))
    run_info = _read_json(os.path.join(exp_dir, "results", "run_info.json"))

    if metrics is None:
        print("Note: exp01_zero_shot/results/metrics.json not found, skipping reference row", file=sys.stderr)
        return None

    num_params = _num_params(run_info)
    return _group_row(
        "exp01_zero_shot (baseline, reused)", "192/12/3", num_params, metrics, run_info, group_key
    )


def _render_table(title, rows):
    lines = [
        f"## {title}\n",
        "| ID | Architecture (embed/depth/heads) | Params | Status | Train loss | Val loss | MAE (norm) | MAE (K) | N eval samples | Eval date |",
        "|----|-----------------------------------|--------|--------|-----------|---------|-----------|---------|-----------------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | {r[9]} |"
        )
    return lines


def main():
    repo_root = _repo_root()
    ablation_dir = os.path.join(
        repo_root, "esdc", "experiments", "exp01_zero_shot", "model_size_ablation"
    )
    results_path = os.path.join(ablation_dir, "results", "RESULTS.md")

    header = (
        "# exp01_zero_shot — Model-Size Ablation Results\n\n"
        "Sub-experiment of `exp01_zero_shot` (see `../README.md` and this "
        "directory's own `README.md`). Same pre-training LMDB, hyperparameters, "
        "and eval dataset as `exp01_zero_shot` — only the ViT architecture "
        "(embed_dim, depth, num_heads) is shrunk. MAE is computed over synthetic "
        "gap pixels in the middle frame only, on the same canonical eval set "
        "(`$ESDC_GAP_FILLING_DIR/dataset.h5`) used by every other experiment in "
        "this repo.\n\n"
        "Unlike `max_gap_ablation`, this ablation studies MAE vs. model "
        "capacity rather than vs. gap-ratio composition, so evaluation is "
        "restricted to the full evaluation dataset (`evaluate_gapfilling.py "
        "--full_only`) — no breakdown by natural gap ratio.\n"
    )

    lines = [header]
    for group_key, title in GROUPS:
        rows = _variant_rows(ablation_dir, group_key)
        ref_row = _reference_row(repo_root, group_key)
        if ref_row is not None:
            rows.insert(0, ref_row)
        lines.extend(_render_table(title, rows))
        lines.append("")

    lines.append(
        "_Regenerate with: `python esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/collect_results.py` "
        "(writes this file directly)._"
    )
    content = "\n".join(lines) + "\n"

    with open(results_path, "w") as f:
        f.write(content)
    print(content, end="")
    print(f"Wrote {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
