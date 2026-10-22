import argparse
import json
import os
import sys

import lmdb

repo_root = os.environ.get("GAP_AWARE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model_gap_aware.lmdb_training import LMDBConsecutiveDataModule, PretrainGapMaskJoinDataModule, build_model
from model_gap_aware.maskfeat import MaskFeatGapFiller
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


_DEFAULT_LMDB = os.environ.get(
    "S2_LMDB_PATH",
    os.path.join(repo_root, "sentinel", "data", "ssl4eo_consecutive_s2c.lmdb"),
)

# Fixed — determined by the Sentinel-2 data format
IMAGE_SIZE = 128
NUM_FRAMES = 5
IN_CHANNELS = 10
TARGET_CHANNELS = list(range(10))


def parse_args():
    p = argparse.ArgumentParser(description="Sentinel-2 gap-aware pre-training")
    p.add_argument("--lmdb_path", default=_DEFAULT_LMDB)
    p.add_argument("--test_lmdb_path", default="")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--prefetch_factor", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap the train+val pool to this many samples (useful for quick runs)")
    p.add_argument("--max_epochs", type=int, default=3)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--log_dir", default=os.path.join(repo_root, "sentinel", "lightning_logs", "pretraining"))
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--decoder_heads", type=int, default=1)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss_type", type=str, default="mae", choices=["mae", "mse"])
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total LR scheduler steps. Defaults to max_epochs * steps_per_epoch, computed from dataset size.",
    )
    p.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * num_gpus * accumulate_grad_batches.",
    )
    p.add_argument(
        "--gap_only_masking",
        action="store_true",
        default=False,
        help="Disable random masking during training; only gap patches are masked. Use for direct gap-filling training.",
    )
    p.add_argument(
        "--structured_masking",
        action="store_true",
        default=False,
        help="Replace random masking with BEiT-style single-block masking of the middle frame.",
    )
    p.add_argument(
        "--gap_mask_lmdb_path",
        type=str,
        default=None,
        help="Path to the lean gap-mask LMDB (gap_masks.lmdb, built by create_gap_mask_lmdb.py). When "
             "provided, uses PretrainGapMaskJoinDataModule instead of LMDBConsecutiveDataModule: (x, y) are "
             "reconstructed on the fly by joining --lmdb_path (and --test_lmdb_path, if present) with this "
             "gap-mask LMDB by idx, enabling direct gap-filling training without duplicating pretraining imagery.",
    )
    p.add_argument(
        "--gap_mask_test_lmdb_path",
        type=str,
        default=None,
        help="Path to the held-out gap-mask LMDB (gap_masks_test.lmdb). Paired with --gap_mask_lmdb_path.",
    )
    p.add_argument(
        "--max_gap_ratio",
        type=float,
        default=None,
        help="Post-hoc filter: drop pre-training samples whose middle-frame band-0 NaN ratio exceeds "
             "this threshold. Band 0 is used as a representative channel -- cloud/shadow masking "
             "zeroes all 10 bands identically at a given pixel (see gee_utils.py), so any single band "
             "carries the same gap ratio. Only supported for the plain pre-training path "
             "(no --gap_mask_lmdb_path).",
    )
    p.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint to fine-tune from. Loads model weights and resets the optimizer "
             "(finetune, not resume). Architecture must match the current config.",
    )
    return p.parse_args()


def _check_lmdb(path: str) -> None:
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    with env.begin(write=False) as txn:
        n = int(txn.stat().get("entries", 0))
        if n == 0:
            raise RuntimeError(f"LMDB has 0 entries: {path}")
        if txn.get(b"0") is None:
            raise RuntimeError(f"LMDB missing key b'0': {path}")
        if txn.get(str(n - 1).encode()) is None:
            raise RuntimeError(f"LMDB missing last key: {path}")
    env.close()


def _check_manifest(lmdb_path: str) -> None:
    manifest_path = lmdb_path + ".manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Rebuild this LMDB with the current "
            "create_LMDB.py / create_gap_mask_lmdb.py to generate it."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest.get("indices"):
        raise RuntimeError(f"Manifest has no indices: {manifest_path}")


def main():
    args = parse_args()

    test_lmdb_path = args.test_lmdb_path or args.lmdb_path.replace(".lmdb", "_test.lmdb")

    if not os.path.exists(args.lmdb_path):
        raise FileNotFoundError(args.lmdb_path)
    _check_lmdb(args.lmdb_path)

    has_test_lmdb = os.path.exists(test_lmdb_path)
    if has_test_lmdb:
        _check_lmdb(test_lmdb_path)

    has_gap_mask_test_lmdb = args.gap_mask_test_lmdb_path is not None and os.path.exists(args.gap_mask_test_lmdb_path)
    if has_gap_mask_test_lmdb:
        _check_lmdb(args.gap_mask_test_lmdb_path)
        _check_manifest(args.gap_mask_test_lmdb_path)

    if args.max_gap_ratio is not None and args.gap_mask_lmdb_path is not None:
        raise ValueError("--max_gap_ratio is only supported for the plain pre-training path (no --gap_mask_lmdb_path).")

    if args.gap_mask_lmdb_path is not None:
        if not os.path.exists(args.gap_mask_lmdb_path):
            raise FileNotFoundError(args.gap_mask_lmdb_path)
        _check_lmdb(args.gap_mask_lmdb_path)
        _check_manifest(args.gap_mask_lmdb_path)
        _check_manifest(args.lmdb_path)
        pretrain_lmdb_paths = [args.lmdb_path]
        if has_test_lmdb:
            _check_manifest(test_lmdb_path)
            pretrain_lmdb_paths.append(test_lmdb_path)
        datamodule = PretrainGapMaskJoinDataModule(
            pretrain_lmdb_paths=pretrain_lmdb_paths,
            gap_mask_lmdb_path=args.gap_mask_lmdb_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            target_channels=TARGET_CHANNELS,
            prefetch_factor=args.prefetch_factor,
            max_samples=args.max_samples,
            test_gap_mask_lmdb_path=args.gap_mask_test_lmdb_path if has_gap_mask_test_lmdb else None,
        )
    else:
        datamodule = LMDBConsecutiveDataModule(
            lmdb_path=args.lmdb_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            target_channels=TARGET_CHANNELS,
            subdir=False,
            prefetch_factor=args.prefetch_factor,
            test_lmdb_path=test_lmdb_path if has_test_lmdb else None,
            max_samples=args.max_samples,
            max_gap_ratio=args.max_gap_ratio,
            gap_ratio_channel=0,
        )

    if args.total_steps is None:
        if args.max_gap_ratio is not None:
            # Raw LMDB entry count doesn't know about the gap_ratio filter applied
            # inside the datamodule -- read the actual post-filter size instead, so
            # the cosine LR schedule's total_steps matches the real number of steps.
            datamodule.setup(stage="fit")
            n_total = len(datamodule.train_ds) + len(datamodule.val_ds)
        else:
            count_lmdb_path = args.gap_mask_lmdb_path if args.gap_mask_lmdb_path is not None else args.lmdb_path
            env = lmdb.open(count_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
            with env.begin(write=False) as txn:
                n_total = int(txn.stat().get("entries", 0))
            env.close()
            if args.max_samples is not None:
                n_total = min(n_total, args.max_samples)
        n_train = n_total - max(1, int(args.val_split * n_total))
        try:
            num_gpus = int(args.devices)
        except (ValueError, TypeError):
            num_gpus = 1
        steps_per_epoch = (n_train // num_gpus) // args.batch_size
        total_steps = steps_per_epoch * args.max_epochs
    else:
        total_steps = args.total_steps
        n_total = None

    if total_steps < args.warmup_steps:
        print(
            f"WARNING: total_steps ({total_steps}) < warmup_steps ({args.warmup_steps}). "
            "The cosine phase will have T_max=0. Reduce --warmup_steps or increase training length."
        )

    if args.pretrained_checkpoint:
        model = MaskFeatGapFiller.load_from_checkpoint(
            args.pretrained_checkpoint,
            lr=args.lr,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            gap_only_masking=args.gap_only_masking,
            structured_masking=args.structured_masking,
            loss_type=args.loss_type,
        )
    else:
        model = build_model(
            image_size=IMAGE_SIZE,
            patch_size=args.patch_size,
            num_frames=NUM_FRAMES,
            in_channels=IN_CHANNELS,
            out_channels=len(TARGET_CHANNELS),
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            decoder_heads=args.decoder_heads,
            lr=args.lr,
            loss_type=args.loss_type,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
            gap_only_masking=args.gap_only_masking,
            structured_masking=args.structured_masking,
        )

    monitor_key = f"{args.loss_type.upper()}_val"
    callbacks = [
        ModelCheckpoint(
            monitor=monitor_key,
            mode="min",
            save_top_k=1,
            filename=f"gapfiller-{{epoch:02d}}-{{{monitor_key}:.4f}}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = TensorBoardLogger(save_dir=args.log_dir, name="")

    os.makedirs(logger.log_dir, exist_ok=True)

    if args.pretrained_checkpoint:
        with open(os.path.join(logger.log_dir, "pretrained_from.json"), "w") as f:
            json.dump({"pretrained_checkpoint": os.path.abspath(args.pretrained_checkpoint)}, f, indent=2)

    training_info = {"max_samples": args.max_samples}
    if n_total is not None:
        training_info["num_finetuning_samples"] = n_total
    with open(os.path.join(logger.log_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        strategy='ddp',
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=datamodule)
    should_test = (
        (args.gap_mask_lmdb_path is None and has_test_lmdb)
        or (args.gap_mask_lmdb_path is not None and has_gap_mask_test_lmdb)
    )
    if should_test:
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
