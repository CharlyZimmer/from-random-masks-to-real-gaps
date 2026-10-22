"""Config-driven launcher for gap_aware_training.py.

Reads an experiment config.yaml, expands ${ENV_VAR} references, and invokes
gap_aware_training.py with the assembled CLI arguments.

All hyperparameters come from config.yaml. Override flags below are for local
sanity-check runs (reduced model/dataset scale) — the Slurm scripts omit them.

Usage:
    # Full HPC run (uses config defaults):
    PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
        --config esdc/experiments/exp04_scratch_gapfilling/config.yaml

    # Local sanity check:
    PYTHONPATH=. python esdc/experiments/scripts/run_training.py \
        --config esdc/experiments/exp04_scratch_gapfilling/config.yaml \
        --accelerator cpu --devices 1 --precision 32 \
        --embed_dim 192 --depth 1 --num_heads 1 --patch_size 8 \
        --max_epochs 2 --max_samples 100
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import yaml

_FALLBACK_PRETRAIN_LOG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "exp01_zero_shot")
)  # only used if config.yaml doesn't set model.pretrained_log_dir


def find_pretrained_checkpoint(checkpoint, log_dir):
    """Resolve the pretrained checkpoint path for finetuning.

    Returns `checkpoint` unchanged if it looks like a real, fully-expanded
    path. Otherwise falls back to auto-discovering the best checkpoint
    (lowest MAE_val in the filename) under `log_dir`, mirroring
    find_checkpoint() in esdc/src/evaluation/evaluate_gapfilling.py.
    """
    if checkpoint and "${" not in checkpoint:
        return checkpoint

    ckpt_files = glob.glob(os.path.join(log_dir, "*", "checkpoints", "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"model.pretrained_checkpoint is unset (ESDC_PRETRAINED_CKPT not exported?) "
            f"and no fallback checkpoint was found under {log_dir}. "
            "Export ESDC_PRETRAINED_CKPT, or run exp01_zero_shot's local train.sh first."
        )

    def _mae_val(path):
        m = re.search(r"MAE_val=([0-9]+(?:\.[0-9]+)?)", os.path.basename(path))
        return float(m.group(1)) if m else float("inf")

    return min(ckpt_files, key=_mae_val)


def _expand(obj):
    """Recursively expand ${ENV_VAR} in all string values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(x) for x in obj]
    return obj


def _deep_merge(base, override):
    """Merge override into base recursively; override wins on all conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Config-driven training launcher")
    p.add_argument("--config", required=True, help="Path to experiment config.yaml")
    # Compute overrides (for local / quick runs)
    p.add_argument("--accelerator", default=None, help="Override training.accelerator")
    p.add_argument("--devices", default=None, help="Override training.devices")
    p.add_argument("--precision", default=None, help="Override training.precision")
    # Model overrides (for local quick-sanity runs with a smaller model)
    p.add_argument("--embed_dim", type=int, default=None, help="Override model.embed_dim")
    p.add_argument("--depth", type=int, default=None, help="Override model.depth")
    p.add_argument("--num_heads", type=int, default=None, help="Override model.num_heads")
    p.add_argument("--patch_size", type=int, default=None, help="Override model.patch_size")
    # Training overrides
    p.add_argument("--max_epochs", type=int, default=None, help="Override training.max_epochs")
    p.add_argument("--max_samples", type=int, default=None, help="Cap dataset size")
    p.add_argument("--batch_size", type=int, default=None, help="Override training.batch_size")
    return p.parse_args()


def main():
    args = parse_args()

    # Load shared config, then deep-merge experiment config on top.
    shared_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "shared_config.yaml")
    )
    with open(shared_path) as f:
        shared_cfg = _expand(yaml.safe_load(f))
    with open(args.config) as f:
        exp_cfg = _expand(yaml.safe_load(f))
    cfg = _deep_merge(shared_cfg, exp_cfg)

    exp_type = cfg["experiment"]["type"]
    # zero_shot runs the pre-training step; the separate evaluate script handles zero-shot eval.

    data = cfg["data"]
    model = cfg["model"]
    t = cfg["training"]

    # Effective values — CLI overrides take precedence over config
    accelerator = args.accelerator or t["accelerator"]
    devices      = args.devices or str(t["devices"])
    precision    = args.precision or t["precision"]
    embed_dim    = args.embed_dim or model["embed_dim"]
    depth        = args.depth or model["depth"]
    num_heads    = args.num_heads or model["num_heads"]
    patch_size   = args.patch_size or model["patch_size"]
    max_epochs   = args.max_epochs or t["max_epochs"]
    batch_size   = args.batch_size or t["batch_size"]

    # LR scaled by number of GPU devices (for DDP)
    try:
        n_devices = int(devices)
    except ValueError:
        n_devices = 1
    lr = t["base_lr"] * n_devices

    in_channels = 1 + len(data["variables"]["aux"])

    cmd = [
        sys.executable, "esdc/src/training/gap_aware_training.py",
        "--lmdb_path",               data["train_lmdb"],
        "--batch_size",              str(batch_size),
        "--num_workers",             str(t["num_workers"]),
        "--val_split",               str(t["val_split"]),
        "--max_epochs",              str(max_epochs),
        "--lr",                      str(lr),
        "--warmup_steps",            str(t["warmup_steps"]),
        "--loss_type",               model["loss_type"],
        "--embed_dim",               str(embed_dim),
        "--depth",                   str(depth),
        "--num_heads",               str(num_heads),
        "--decoder_heads",           str(model["decoder_heads"]),
        "--patch_size",              str(patch_size),
        "--in_channels",             str(in_channels),
        "--accelerator",             accelerator,
        "--devices",                 devices,
        "--precision",               precision,
        "--log_dir",                 t["log_dir"],
        "--accumulate_grad_batches", str(t.get("accumulate_grad_batches", 1)),
    ]

    if "train_lmdb_y" in data:
        cmd.extend(["--y_lmdb_path", data["train_lmdb_y"]])
    if "test_lmdb" in data:
        cmd.extend(["--test_lmdb_path", data["test_lmdb"]])
    if "test_lmdb_y" in data:
        cmd.extend(["--test_y_lmdb_path", data["test_lmdb_y"]])
    if "max_gap_ratio" in data:
        cmd.extend(["--max_gap_ratio", str(data["max_gap_ratio"])])
    if t["gap_only_masking"]:
        cmd.append("--gap_only_masking")
    if t.get("structured_masking"):
        cmd.append("--structured_masking")
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if exp_type == "finetuning":
        log_dir = model.get("pretrained_log_dir") or _FALLBACK_PRETRAIN_LOG_DIR
        ckpt_path = find_pretrained_checkpoint(model.get("pretrained_checkpoint"), log_dir)
        cmd.extend(["--pretrained_checkpoint", ckpt_path])

    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
