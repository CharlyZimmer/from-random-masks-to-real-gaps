"""Plot eval loss vs. model size for the model-size ablation.

Reads model_size_ablation/results/<label>/{metrics.json,run_info.json} for
each config in configs/*.yaml, plus the exp01_zero_shot baseline's own
results/ (the 192/12/3 point, not retrained here — see README.md), and plots
mean_mae_kelvin (y) against parameter count (x, log scale, read from each
run_info.json's model.num_params). Each point is labeled with its
embed_dim/depth/num_heads configuration. Saves a PNG.

Usage (from project root):
    python esdc/experiments/exp01_zero_shot/model_size_ablation/scripts/plot_results.py
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import yaml

# Formal print style, matching the paper's default Latin Modern/Computer Modern
# serif body (no fontspec is used in main.tex, so this is the LaTeX default).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

_INK = "#1a1a1a"        # axis labels, spines, tick text
_INK_MUTED = "#595959"  # tick labels
_GRIDLINE = "#d9d9d9"   # hairline horizontal gridlines
_LINE = "#1f5fa8"       # main series


def _repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _variant_labels(ablation_dir):
    configs_dir = os.path.join(ablation_dir, "configs")
    labels = []
    for name in sorted(os.listdir(configs_dir)):
        if name.endswith(".yaml"):
            labels.append(name[: -len(".yaml")])
    return labels


def _point(results_dir, label, arch_label):
    metrics = _read_json(os.path.join(results_dir, label, "metrics.json"))
    run_info = _read_json(os.path.join(results_dir, label, "run_info.json"))
    if metrics is None or run_info is None:
        return None
    num_params = (run_info.get("model") or {}).get("num_params")
    if num_params is None:
        return None
    return (num_params, metrics["mean_mae_kelvin"], arch_label)


def _baseline_point(repo_root):
    exp_dir = os.path.join(repo_root, "esdc", "experiments", "exp01_zero_shot")
    metrics = _read_json(os.path.join(exp_dir, "results", "metrics.json"))
    run_info = _read_json(os.path.join(exp_dir, "results", "run_info.json"))
    if metrics is None or run_info is None:
        return None
    num_params = (run_info.get("model") or {}).get("num_params")
    if num_params is None:
        return None
    return (num_params, metrics["mean_mae_kelvin"], "192/12/3")


def main():
    repo_root = _repo_root()
    ablation_dir = os.path.join(
        repo_root, "esdc", "experiments", "exp01_zero_shot", "model_size_ablation"
    )
    default_results_dir = os.path.join(ablation_dir, "results")

    p = argparse.ArgumentParser(description="Plot eval loss vs. model size")
    p.add_argument("--results_dir", default=default_results_dir)
    p.add_argument("--out", default=None, help="Defaults to <results_dir>/loss_vs_model_size.png")
    args = p.parse_args()

    out_path = args.out or os.path.join(args.results_dir, "loss_vs_model_size.png")

    configs_dir = os.path.join(ablation_dir, "configs")
    labels = _variant_labels(ablation_dir)

    points = []
    for label in labels:
        with open(os.path.join(configs_dir, f"{label}.yaml")) as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        arch_label = f"{m['embed_dim']}/{m['depth']}/{m['num_heads']}"
        pt = _point(args.results_dir, label, arch_label)
        if pt is not None:
            points.append(pt)

    baseline = _baseline_point(repo_root)
    if baseline is not None:
        points.append(baseline)

    points.sort(key=lambda p_: p_[0])

    if len(points) < 2:
        n_variants = len(labels) + (1 if baseline is not None else 0)
        print(
            f"Not enough data points yet ({len(points)} of {n_variants} variants evaluated) — "
            "run scripts/slurm/evaluate_all.sh on the HPC first. Skipping plot.",
            file=sys.stderr,
        )
        return

    xs = [p_[0] for p_ in points]
    ys = [p_[1] for p_ in points]
    arch_labels = [p_[2] for p_ in points]

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)

    # Recessive horizontal gridlines only, hairline, solid, behind everything.
    ax.grid(axis="y", color=_GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_INK)

    ax.plot(
        xs, ys,
        color=_LINE, linewidth=2, solid_joinstyle="round", solid_capstyle="round",
        marker="o", markersize=8, markerfacecolor=_LINE,
        markeredgecolor="white", markeredgewidth=2,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: arch_labels[xs.index(v)] if v in xs else f"{int(v):,}"
    ))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="both", colors=_INK_MUTED, labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Encoder configuration (embedding dim / depth / heads)", color=_INK, fontsize=10)
    ax.set_ylabel("Evaluation MAE (K)", color=_INK, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
