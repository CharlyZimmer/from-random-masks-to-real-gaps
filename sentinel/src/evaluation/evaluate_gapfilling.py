"""Zero-shot gap-filling evaluation on the held-out gap-filling dataset.

Loads a pre-trained MaskFeatGapFiller checkpoint, runs inference with
gap_only_masking=True (no random masking — only NaN-containing patches are
masked), and reports pixel-accurate MAE restricted to the synthetic gap pixels
from gap_mask.tif.

Run from project root:
    PYTHONPATH=. python sentinel/src/evaluation/evaluate_gapfilling.py \
        --data_dir $S2_DATA_DIR/gap_filling --device cpu
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset, Subset

repo_root = os.environ.get(
    "GAP_AWARE_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", ".."),
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model_gap_aware.maskfeat import MaskFeatGapFiller
from sentinel.src.dataset.create_LMDB import S2ConsecutiveDataset, _sentinel2_normalize

_DEFAULT_DATA_DIR = os.path.join(
    os.environ.get("S2_DATA_DIR", "./sentinel/data"),
    "gap_filling",
)
_DEFAULT_LOG_DIR = "./sentinel/lightning_logs/pretraining"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GapFillingEvalDataset(Dataset):
    """Loads (input_series, gt_middle, gap_mask) from the gap-filling folder.

    Returns:
        input_series: (T, C, H, W) float32, NaN where synthetic gaps were injected
        gt_middle:    (C, H, W) float32, original clean middle frame
        gap_mask:     (H, W) float32, 1.0 = synthetic gap pixel
    """

    def __init__(self, data_dir: str, normalize: bool = False) -> None:
        self.gt_dir = os.path.join(data_dir, "ground_truth")
        self.normalize = normalize
        self._input_ds = S2ConsecutiveDataset(
            root=data_dir,
            subdir="SSL-consecutive",
            normalize=normalize,
        )
        self.locations = self._input_ds.locations

    def __len__(self) -> int:
        return len(self._input_ds)

    def __getitem__(self, idx: int):
        input_series = self._input_ds[idx]  # (T, C, H, W) numpy, NaN for gaps

        loc_id = self.locations[idx]
        gt_loc = os.path.join(self.gt_dir, loc_id)

        ts_dirs = sorted(
            d for d in os.listdir(gt_loc)
            if os.path.isdir(os.path.join(gt_loc, d))
        )
        if not ts_dirs:
            raise RuntimeError(f"No timestamp dirs in {gt_loc}")
        gt_ts_dir = os.path.join(gt_loc, ts_dirs[0])

        # Read GT multi-band TIF — no zero-to-NaN conversion
        with rasterio.open(os.path.join(gt_ts_dir, "S2C.tif")) as src:
            gt_middle = src.read().astype(np.float32)  # (C, H, W)

        if self.normalize:
            gt_middle = _sentinel2_normalize(gt_middle)

        # Read gap mask
        with rasterio.open(os.path.join(gt_ts_dir, "gap_mask.tif")) as src:
            gap_mask = src.read(1).astype(np.float32)  # (H, W), 1=gap

        return input_series, gt_middle, gap_mask


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

    best = min(ckpt_files, key=_mae_val)
    return best


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: MaskFeatGapFiller, dataloader: DataLoader, device: torch.device):
    model.eval()

    C = None
    per_band_sum = None
    per_band_count = None
    gap_fractions = []

    for batch in dataloader:
        x, gt, gap_mask = batch
        x = x.to(device)
        gt = gt.to(device)
        gap_mask = gap_mask.to(device)

        pred, _ = model(x, gap_only_masking=True)  # (B, C, H, W)

        if C is None:
            C = pred.shape[1]
            per_band_sum   = torch.zeros(C, device=device)
            per_band_count = torch.zeros(C, device=device)

        # Restrict to gap pixels with valid (non-NaN) GT.
        # GT can have NaN at image-edge pixels from the GEE download; those must
        # be excluded so they don't poison the accumulator.
        gt_valid = ~torch.isnan(gt)              # (B, C, H, W)
        gt = torch.nan_to_num(gt, nan=0.0)

        gap_mask_4d = gap_mask.unsqueeze(1).bool()          # (B, 1, H, W)
        loss_mask = (gap_mask_4d & gt_valid).float()        # (B, C, H, W)

        err = torch.abs(pred - gt)
        masked_err = err * loss_mask

        per_band_sum   += masked_err.sum(dim=(0, 2, 3))       # (C,)
        per_band_count += loss_mask.sum(dim=(0, 2, 3))        # (C,)

        B, H, W = gap_mask.shape
        gap_fractions.extend((gap_mask.sum(dim=(1, 2)) / (H * W)).cpu().tolist())

    if per_band_count is None or per_band_count.sum() == 0:
        print("WARNING: no valid gap pixels found across all samples.")
        return None

    per_band_mae = (per_band_sum / per_band_count.clamp(min=1)).cpu()
    mean_mae = per_band_mae.mean().item()

    return {
        "mean_mae": mean_mae,
        "per_band_mae": per_band_mae.tolist(),
        "n_samples": len(gap_fractions),
        "gap_coverage_mean": float(np.mean(gap_fractions)),
        "gap_coverage_std": float(np.std(gap_fractions)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot gap-filling evaluation")
    p.add_argument("--data_dir", default=_DEFAULT_DATA_DIR,
                   help="Path to gap_filling/ directory")
    p.add_argument("--log_dir", default=_DEFAULT_LOG_DIR,
                   help="Lightning logs directory for checkpoint discovery")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint path (overrides auto-discovery)")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Max samples to evaluate (default: all)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--normalize", action="store_true",
                   help="Apply Sentinel-2 normalization (÷10000, clip [0,1]) — must match training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default=None,
                   help="Directory to write metrics.json and run_info.json (for tracked experiment runs)")
    p.add_argument("--experiment_id", default=None,
                   help="Experiment identifier written into run_info.json")
    p.add_argument("--training_metrics_only", "--training-metrics-only",
                   dest="training_metrics_only", action="store_true",
                   help="Skip model inference entirely. Read the best-epoch "
                        "training loss and validation loss from the checkpoint "
                        "and its TensorBoard event files and merge them into an "
                        "existing results/metrics.json. Requires --results_dir "
                        "with a run_info.json; aborts unless the resolved "
                        "checkpoint matches the one that evaluation used.")
    return p.parse_args()


def _write_results(args, results, ckpt_path, model):
    """Write metrics.json and run_info.json to --results_dir when provided.

    Sentinel has no single physical unit like ESDC's Kelvin range (10 bands,
    each with its own reflectance scale), so only normalized/raw-DN MAE is
    reported — no Kelvin-equivalent rescale.
    """
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

    # Best-epoch train/val loss from the checkpoint + its event files, so a full
    # re-eval keeps the results-table columns populated. Non-fatal: a missing
    # event file just means those columns stay empty for this run.
    from model_gap_aware.checkpoint_metrics import training_loss_metrics
    try:
        metrics.update(training_loss_metrics(ckpt_path, kelvin=False))
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
        payload = training_loss_metrics(ckpt_path, kelvin=False)
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

    dataset = GapFillingEvalDataset(args.data_dir, normalize=args.normalize)
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
    results = evaluate(model, dataloader, device)

    if results is None:
        return

    print("\n=== Gap-Filling Evaluation Results ===")
    print(f"Samples evaluated : {results['n_samples']}")
    print(f"Gap coverage      : {results['gap_coverage_mean']*100:.1f}% ± {results['gap_coverage_std']*100:.1f}%")
    print(f"Mean MAE (gap px) : {results['mean_mae']:.4f}")
    print("Per-band MAE:")
    for i, mae in enumerate(results["per_band_mae"]):
        print(f"  Band {i}: {mae:.4f}")

    _write_results(args, results, ckpt_path, model)


if __name__ == "__main__":
    main()
