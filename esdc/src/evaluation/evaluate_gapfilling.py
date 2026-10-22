"""Zero-shot gap-filling evaluation on the held-out ESDC gap-filling dataset.

Loads a pre-trained MaskFeatGapFiller checkpoint, runs inference with
gap_only_masking=True (no random masking — only NaN-containing patches are
masked), and reports pixel-accurate MAE restricted to the synthetic gap pixels.

The gap-filling dataset is created with --syn_gaps --no_split, so all samples
are stored under X_train/y_train. y_train holds the clean ground-truth LST
only at gap pixels (NaN elsewhere); the gap mask is implicit: a pixel is a
gap pixel iff y_train is not NaN there.

ESDC values are min-max normalized to [0, 1] during dataset creation
(patchfill/preprocessing.py). MAE is therefore reported in [0, 1] units.
The normalization scale is read from meta.json in the dataset directory.

Run from project root:
    PYTHONPATH=. python esdc/src/evaluation/evaluate_gapfilling.py \
        --data_dir esdc/data/lst_gap_filling/dataset.h5 --device cpu
"""
import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

repo_root = os.environ.get(
    "GAP_AWARE_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", ".."),
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model_gap_aware.maskfeat import MaskFeatGapFiller

_DEFAULT_H5 = os.path.join(
    os.environ.get("ESDC_GAP_FILLING_DIR", "./esdc/data/lst_gap_filling"),
    "dataset.h5",
)
_DEFAULT_LOG_DIR = os.path.join(repo_root, "esdc", "lightning_logs", "pretraining")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ESDCGapFillingEvalDataset(Dataset):
    """Loads (input_series, gt_middle, gap_mask) from the ESDC gap-filling HDF5.

    Returns:
        input_series: (T, C, H, W) float32 — model input; middle frame has NaN at gaps
        gt_middle:    (1, H, W) float32 — clean LST at gap pixels, NaN elsewhere
        gap_mask:     (H, W) float32 — 1.0 = synthetic gap pixel
    """

    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path
        self._file = None
        with h5py.File(h5_path, "r") as f:
            self._length = len(f["X_train"])

    def __len__(self) -> int:
        return self._length

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __getitem__(self, idx: int):
        f = self._get_file()
        x_np = f["X_train"][idx]   # (T, H, W, C) channels-last
        y_np = f["y_train"][idx]   # (1, H, W, 1) channels-last

        # (T, H, W, C) → (T, C, H, W)
        x = torch.from_numpy(
            np.ascontiguousarray(np.transpose(x_np, (0, 3, 1, 2))).astype(np.float32)
        )

        gt = y_np[0, :, :, 0].astype(np.float32)              # (H, W)
        gap_mask = (~np.isnan(gt)).astype(np.float32)          # (H, W), 1=gap pixel
        gt_middle = gt[np.newaxis, :, :]                       # (1, H, W)

        return x, torch.from_numpy(gt_middle), torch.from_numpy(gap_mask)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoint(log_dir: str, checkpoint: str | None = None) -> str:
    if checkpoint:
        return checkpoint

    pattern = os.path.join(log_dir, "*", "checkpoints", "*.ckpt")
    ckpt_files = glob.glob(pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found under {log_dir}")

    def _mae_val(path: str) -> float:
        m = re.search(r"MAE_val=([0-9]+(?:\.[0-9]+)?)", os.path.basename(path))
        return float(m.group(1)) if m else float("inf")

    return min(ckpt_files, key=_mae_val)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

# Groups samples by their natural (pre-donor-injection) gap fraction in the
# target patch -- i.e. the quantity max_gap_portion bounds at dataset-creation
# time (patchfill/dataset_building.py), NOT the donor-mask-driven fraction of
# gap_mask (which is what the loss itself is computed over, unchanged, and is
# governed separately by gap_donor_min_ratio/gap_donor_max_ratio).
# None means "no filter" -- the "full" group covers every sample.
GROUP_THRESHOLDS = {"full": None, "max_0.10": 0.10, "max_0.00": 0.0}


def _empty_accumulators():
    return {
        "per_band_sum": None,
        "per_band_count": None,
        "gap_fractions": [],
    }


@torch.no_grad()
def evaluate(
    model: MaskFeatGapFiller,
    dataloader: DataLoader,
    device: torch.device,
    group_thresholds: dict = GROUP_THRESHOLDS,
):
    model.eval()

    C = None
    groups = {key: _empty_accumulators() for key in group_thresholds}

    for batch in dataloader:
        x, gt, gap_mask = batch
        x = x.to(device)
        gt = gt.to(device)
        gap_mask = gap_mask.to(device)

        # Natural, pre-injection gap fraction per sample -- computed before gt
        # is nan_to_num'd below. A pixel was naturally missing iff it is NaN
        # in both the (post-injection) input's middle frame and the target
        # (see plan / visualize_gapfilling_samples.ipynb for the derivation).
        T = x.shape[1]
        raw_nan = torch.isnan(x[:, T // 2, 0]) & torch.isnan(gt[:, 0])  # (B, H, W)
        raw_gap_fraction = raw_nan.float().mean(dim=(1, 2))            # (B,)

        pred, _ = model(x, gap_only_masking=True)  # (B, C, H, W)

        if C is None:
            C = pred.shape[1]
            for g in groups.values():
                g["per_band_sum"]   = torch.zeros(C, device=device)
                g["per_band_count"] = torch.zeros(C, device=device)

        # Restrict to gap pixels with valid (non-NaN) GT.
        gt_valid = ~torch.isnan(gt)           # (B, C, H, W)
        gt = torch.nan_to_num(gt, nan=0.0)

        gap_mask_4d = gap_mask.unsqueeze(1).bool()        # (B, 1, H, W)
        loss_mask = (gap_mask_4d & gt_valid).float()      # (B, C, H, W)

        err = torch.abs(pred - gt)
        masked_err = err * loss_mask

        B, H, W = gap_mask.shape
        donor_gap_fractions = gap_mask.sum(dim=(1, 2)) / (H * W)  # (B,)

        for key, thresh in group_thresholds.items():
            sel = torch.ones(B, dtype=torch.bool, device=device) if thresh is None \
                else (raw_gap_fraction <= thresh)
            if not sel.any():
                continue
            g = groups[key]
            g["per_band_sum"]   += masked_err[sel].sum(dim=(0, 2, 3))   # (C,)
            g["per_band_count"] += loss_mask[sel].sum(dim=(0, 2, 3))    # (C,)
            g["gap_fractions"].extend(donor_gap_fractions[sel].cpu().tolist())

    if C is None:
        print("WARNING: no valid gap pixels found across all samples.")
        return None

    results = {}
    for key, g in groups.items():
        n = len(g["gap_fractions"])
        if n == 0 or g["per_band_count"].sum() == 0:
            print(f"Note: gap ratio group '{key}' has no eligible samples with valid gap pixels.")
            results[key] = {
                "mean_mae": None,
                "per_band_mae": None,
                "n_samples": n,
                "gap_coverage_mean": None,
                "gap_coverage_std": None,
            }
            continue

        per_band_mae = (g["per_band_sum"] / g["per_band_count"].clamp(min=1)).cpu()
        results[key] = {
            "mean_mae": per_band_mae.mean().item(),
            "per_band_mae": per_band_mae.tolist(),
            "n_samples": n,
            "gap_coverage_mean": float(np.mean(g["gap_fractions"])),
            "gap_coverage_std": float(np.std(g["gap_fractions"])),
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot ESDC gap-filling evaluation")
    p.add_argument("--data_dir", default=_DEFAULT_H5,
                   help="Path to the gap-filling dataset.h5 file")
    p.add_argument("--log_dir", default=_DEFAULT_LOG_DIR,
                   help="Lightning logs directory for checkpoint discovery")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint path (overrides auto-discovery)")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Max samples to evaluate (default: all)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default=None,
                   help="Directory to write metrics.json and run_info.json (for tracked experiment runs)")
    p.add_argument("--experiment_id", default=None,
                   help="Experiment identifier written into run_info.json")
    p.add_argument("--full_only", action="store_true",
                   help="Only evaluate/report the full-dataset group; skip the "
                        "natural-gap-ratio breakdown and omit gap_ratio_groups "
                        "from metrics.json")
    p.add_argument("--training_metrics_only", "--training-metrics-only",
                   dest="training_metrics_only", action="store_true",
                   help="Skip model inference entirely. Read the best-epoch "
                        "training loss and validation loss from the checkpoint "
                        "and its TensorBoard event files and merge them into an "
                        "existing results/metrics.json. Requires --results_dir "
                        "with a run_info.json; aborts unless the resolved "
                        "checkpoint matches the one that evaluation used.")
    return p.parse_args()


def _load_norm_range(data_dir: str):
    """Read meta.json from data_dir and return the normalization range used."""
    meta_path = os.path.join(data_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get('valid_range') or meta.get('computed_range')


def _group_metrics(group_results, scale):
    """Build the metrics.json sub-dict for one gap ratio group."""
    if group_results["mean_mae"] is None:
        return {
            "mean_mae_normalized": None,
            "mean_mae_kelvin": None,
            "per_band_mae_normalized": None,
            "per_band_mae_kelvin": None,
            "n_samples": group_results["n_samples"],
            "gap_coverage_mean": None,
            "gap_coverage_std": None,
        }
    return {
        "mean_mae_normalized": group_results["mean_mae"],
        "mean_mae_kelvin": round(group_results["mean_mae"] * scale, 4) if scale else None,
        "per_band_mae_normalized": group_results["per_band_mae"],
        "per_band_mae_kelvin": [round(v * scale, 4) for v in group_results["per_band_mae"]] if scale else None,
        "n_samples": group_results["n_samples"],
        "gap_coverage_mean": group_results["gap_coverage_mean"],
        "gap_coverage_std": group_results["gap_coverage_std"],
    }


def _write_results(args, results, norm_range, ckpt_path, model):
    """Write metrics.json and run_info.json to --results_dir when provided.

    `results` is the dict returned by evaluate(): one entry per gap ratio
    group ("full", "max_0.10", "max_0.00"). Top-level metrics.json keys are
    sourced from the "full" group so existing consumers (other experiments'
    collect_results.py) keep working unchanged; the full per-group breakdown
    is added under "gap_ratio_groups".
    """
    if args.results_dir is None:
        return

    import datetime
    import subprocess as _sp

    os.makedirs(args.results_dir, exist_ok=True)
    scale = (norm_range[1] - norm_range[0]) if norm_range is not None else None
    full = results["full"]

    metrics = {
        "mean_mae_normalized": full["mean_mae"],
        "mean_mae_kelvin": round(full["mean_mae"] * scale, 4) if scale and full["mean_mae"] is not None else None,
        "per_band_mae_normalized": full["per_band_mae"],
        "per_band_mae_kelvin": [round(v * scale, 4) for v in full["per_band_mae"]] if scale and full["per_band_mae"] else None,
        "n_samples": full["n_samples"],
        "gap_coverage_mean": full["gap_coverage_mean"],
        "gap_coverage_std": full["gap_coverage_std"],
        "norm_range": list(norm_range) if norm_range else None,
    }
    if not args.full_only:
        metrics["gap_ratio_groups"] = {key: _group_metrics(g, scale) for key, g in results.items()}

    # Best-epoch train/val loss from the checkpoint + its event files, so a full
    # re-eval keeps the results-table columns populated. Non-fatal: a missing
    # event file just means those columns stay empty for this run.
    from model_gap_aware.checkpoint_metrics import training_loss_metrics
    try:
        metrics.update(training_loss_metrics(ckpt_path, kelvin=scale is not None))
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read training/val loss from logs: {e}")

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

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # If this checkpoint was produced by fine-tuning from another checkpoint
    # (see gap_aware_training.py), surface that source path so it's easy to
    # confirm which pretrained model an experiment actually built on.
    pretrained_checkpoint_path = None
    version_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    pretrained_from_path = os.path.join(version_dir, "pretrained_from.json")
    if os.path.exists(pretrained_from_path):
        with open(pretrained_from_path) as f:
            pretrained_checkpoint_path = json.load(f).get("pretrained_checkpoint")

    # Number of samples the checkpoint was fine-tuned on (see training_info.json,
    # written by gap_aware_training.py).
    num_finetuning_samples = None
    training_info_path = os.path.join(version_dir, "training_info.json")
    if os.path.exists(training_info_path):
        with open(training_info_path) as f:
            num_finetuning_samples = json.load(f).get("num_finetuning_samples")

    run_info = {
        "experiment_id": exp_id,
        "checkpoint_path": os.path.abspath(ckpt_path),
        "pretrained_checkpoint_path": pretrained_checkpoint_path,
        "num_finetuning_samples": num_finetuning_samples,
        "eval_dataset_path": os.path.abspath(args.data_dir),
        "eval_date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "model": {
            "num_params": num_params,
            "num_trainable_params": num_trainable_params,
            "embed_dim": model.hparams.get("embed_dim"),
            "depth": model.hparams.get("depth"),
            "num_heads": model.hparams.get("heads"),
            "decoder_heads": model.hparams.get("decoder_heads"),
            "patch_size": model.hparams.get("patch_size"),
            "image_size": model.hparams.get("image_size"),
            "num_frames": model.hparams.get("num_frames"),
            "in_channels": model.hparams.get("in_channels"),
            "out_channels": model.hparams.get("out_channels"),
        },
    }
    run_info_path = os.path.join(args.results_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info written to: {run_info_path}")


def _run_training_metrics_only(args):
    """Merge best-epoch train/val loss into an existing results/metrics.json.

    No model inference: only the checkpoint (val loss) and its TensorBoard event
    files (train loss) are read. Aborts unless the resolved checkpoint is the
    same one the committed evaluation used, per run_info.json.
    """
    from model_gap_aware.checkpoint_metrics import (
        assert_same_checkpoint,
        checkpoint_identity,
        training_loss_metrics,
    )

    if not args.results_dir:
        raise SystemExit("ERROR: --training_metrics_only requires --results_dir")

    metrics_path = os.path.join(args.results_dir, "metrics.json")
    run_info_path = os.path.join(args.results_dir, "run_info.json")
    if not os.path.exists(metrics_path):
        raise SystemExit(f"ERROR: {metrics_path} not found — run the full evaluation first")
    if not os.path.exists(run_info_path):
        raise SystemExit(f"ERROR: {run_info_path} not found — cannot verify checkpoint identity")

    with open(run_info_path) as f:
        recorded_ckpt = json.load(f).get("checkpoint_path")
    if not recorded_ckpt:
        raise SystemExit(f"ERROR: no checkpoint_path recorded in {run_info_path}")

    ckpt_path = find_checkpoint(args.log_dir, args.checkpoint)
    assert_same_checkpoint(ckpt_path, recorded_ckpt)
    print(f"Checkpoint confirmed against run_info.json: {'/'.join(checkpoint_identity(ckpt_path))}")

    try:
        payload = training_loss_metrics(ckpt_path, kelvin=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"ERROR: could not read training/val loss from the Lightning logs: {e}")

    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics.update(payload)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMerged into {metrics_path}:")
    for k, v in payload.items():
        print(f"  {k}: {v}")


def main():
    args = parse_args()

    if args.training_metrics_only:
        _run_training_metrics_only(args)
        return

    ckpt_path = find_checkpoint(args.log_dir, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    model = MaskFeatGapFiller.load_from_checkpoint(ckpt_path, map_location=args.device)
    model.to(args.device)
    model.eval()

    dataset = ESDCGapFillingEvalDataset(args.data_dir)
    print(f"Dataset size: {len(dataset)} samples")

    if args.num_samples is not None and args.num_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), size=args.num_samples, replace=False).tolist()
        dataset = Subset(dataset, indices)
        print(f"Evaluating on {args.num_samples} randomly selected samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Running inference...")
    device = torch.device(args.device)
    group_thresholds = {"full": None} if args.full_only else GROUP_THRESHOLDS
    results = evaluate(model, dataloader, device, group_thresholds=group_thresholds)

    if results is None:
        return

    norm_range = _load_norm_range(os.path.dirname(os.path.abspath(args.data_dir)))
    scale = (norm_range[1] - norm_range[0]) if norm_range is not None else None

    group_labels = {
        "full": "Full dataset",
        "max_0.10": "≤ 10% natural gap ratio",
        "max_0.00": "0% natural gap ratio (fully clean target)",
    }

    print("\n=== Gap-Filling Evaluation Results ===")
    for key, g in results.items():
        label = group_labels.get(key, key)
        print(f"\n--- {label} ---")
        print(f"Samples evaluated : {g['n_samples']}")
        if g["mean_mae"] is None:
            print("No eligible samples with valid gap pixels in this group.")
            continue
        print(f"Gap coverage      : {g['gap_coverage_mean']*100:.1f}% ± {g['gap_coverage_std']*100:.1f}%")
        if scale is not None:
            print(f"Mean MAE (gap px) : {g['mean_mae']:.4f}  [{g['mean_mae']*scale:.2f} K]")
        else:
            print(f"Mean MAE (gap px) : {g['mean_mae']:.4f}")
        print("Per-band MAE:")
        for i, mae in enumerate(g["per_band_mae"]):
            if scale is not None:
                print(f"  Band {i} (LST): {mae:.4f}  [{mae*scale:.2f} K]")
            else:
                print(f"  Band {i} (LST): {mae:.4f}")

    _write_results(args, results, norm_range, ckpt_path, model)


if __name__ == "__main__":
    main()
