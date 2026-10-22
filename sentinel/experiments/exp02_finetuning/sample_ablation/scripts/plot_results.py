"""Plot eval loss vs. number of fine-tuning samples for the sample-count ablation.

Reads sample_ablation/results/n*/metrics.json, plots mean_mae_normalized (y) against
the fine-tuning sample count (x, log scale), and overlays exp04_scratch_gapfilling
as a muted reference line. Saves a PNG.

Sentinel port of esdc/experiments/exp02_finetuning/sample_ablation/scripts/plot_results.py:
plots mean_mae_normalized instead of mean_mae_kelvin -- Sentinel has no single physical
unit across its 10 bands.

Usage (from project root):
    python sentinel/experiments/exp02_finetuning/sample_ablation/scripts/plot_results.py
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

_LABEL_RE = re.compile(r"^n(\d+)$")

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["Liberation Mono", "DejaVu Sans Mono"],
    "axes.unicode_minus": False,
})

_INK = "#1a1a1a"        # header, axis labels, legend text
_INK_SOFT = "#3d3d3d"   # tick labels
_INK_MUTED = "#6b6b6b"  # reference/baseline line
_GRIDLINE = "#e6e6e6"   # hairline horizontal gridlines
_LINE = "#6527F5"       # main series


def _format_samples(v, _pos):
    v = int(v)
    return f"{v / 1000:g}k" if v >= 1000 else str(v)


def _repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _collect_points(results_dir):
    points = []
    for name in sorted(os.listdir(results_dir)):
        m = _LABEL_RE.match(name)
        if not m:
            continue
        metrics = _read_json(os.path.join(results_dir, name, "metrics.json"))
        if metrics is None:
            continue
        points.append((int(m.group(1)), metrics["mean_mae_normalized"]))
    points.sort(key=lambda p: p[0])
    return points


def _count_variants(results_dir):
    return sum(1 for name in os.listdir(results_dir) if _LABEL_RE.match(name))


def _reference_value(repo_root, exp_dir_name):
    metrics = _read_json(
        os.path.join(repo_root, "sentinel", "experiments", exp_dir_name, "results", "metrics.json")
    )
    return metrics["mean_mae_normalized"] if metrics is not None else None


def main():
    repo_root = _repo_root()
    default_results_dir = os.path.join(
        repo_root, "sentinel", "experiments", "exp02_finetuning", "sample_ablation", "results"
    )

    p = argparse.ArgumentParser(description="Plot eval loss vs. fine-tuning sample count")
    p.add_argument("--results_dir", default=default_results_dir)
    p.add_argument("--out", default=None, help="Defaults to <results_dir>/loss_vs_samples.png")
    args = p.parse_args()

    out_path = args.out or os.path.join(args.results_dir, "loss_vs_samples.png")

    points = _collect_points(args.results_dir)
    if len(points) < 2:
        n_variants = _count_variants(args.results_dir)
        print(
            f"Not enough data points yet ({len(points)} of {n_variants} variants evaluated) — "
            "run scripts/slurm/evaluate_n*.sh on the HPC first. Skipping plot.",
            file=sys.stderr,
        )
        return

    xs = [p_[0] for p_ in points]
    ys = [p_[1] for p_ in points]

    ref_exp04 = _reference_value(repo_root, "exp04_scratch_gapfilling")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    # No spines, no tick marks; recessive horizontal gridlines only.
    ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Main series: the ablation sweep.
    ax.plot(
        xs, ys,
        color=_LINE, linewidth=2.75, solid_joinstyle="round", solid_capstyle="round",
        marker="o", markersize=9, markerfacecolor=_LINE,
        markeredgecolor="white", markeredgewidth=2,
        zorder=3, label="Fine-tuned (pretrained)",
    )

    # Reference line: muted, dashed.
    if ref_exp04 is not None:
        ax.axhline(
            ref_exp04, color=_INK_MUTED, linewidth=1.75, linestyle=(0, (3, 2)),
            zorder=2, label="From-scratch baseline",
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_samples))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="both", length=0, pad=8, colors=_INK_SOFT, labelsize=12)

    # Headroom so a baseline near the data extremes doesn't hug the plot edge,
    # then a soft fill under the sweep down to the (fixed) lower limit.
    ax.margins(y=0.08)
    y_lo, y_hi = ax.get_ylim()
    ax.fill_between(xs, ys, y_lo, color=_LINE, alpha=0.08, linewidth=0, zorder=1)
    ax.set_ylim(y_lo, y_hi)

    # Y-axis title as a horizontal header, legend on the row beneath it --
    # both outside the axes so no data line can obstruct them.
    ax.text(
        0, 1.13, "(b) Sentinel-2: Evaluation MAE (normalized)", transform=ax.transAxes,
        ha="left", va="bottom", color=_INK, fontsize=14, fontweight="bold",
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
        fontsize=12, labelcolor=_INK, handlelength=2.2, columnspacing=2.0,
        borderaxespad=0.3,
    )
    ax.set_xlabel("Number of fine-tuning samples", color=_INK, fontsize=14, labelpad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
