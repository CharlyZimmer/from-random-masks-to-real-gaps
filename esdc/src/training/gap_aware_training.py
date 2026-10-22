import argparse
import json
import os
import sys

import lmdb

repo_root = os.environ.get("GAP_AWARE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model_gap_aware.lmdb_training import GapFillingTrainingDataModule, LMDBConsecutiveDataModule, build_model
from model_gap_aware.maskfeat import MaskFeatGapFiller
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


_esdc_data = os.environ.get(
    "ESDC_DATA_DIR",
    os.path.join(repo_root, "esdc", "data", "lst_no_syn_gaps"),
)
_DEFAULT_LMDB = os.path.join(_esdc_data, "lmdb", "dataset.lmdb")
_DEFAULT_TEST_LMDB = os.path.join(_esdc_data, "lmdb", "dataset_test.lmdb")
_DEFAULT_TEST_Y_LMDB = os.path.join(_esdc_data, "lmdb", "dataset_test_y.lmdb")

TARGET_CHANNELS = [0] # reconstruct LST only

# Derived from meta.json at runtime (see main())
IMAGE_SIZE = None
NUM_FRAMES = None
IN_CHANNELS = None


def parse_args():
    p = argparse.ArgumentParser(description="ESDC gap-aware pre-training")
    p.add_argument("--lmdb_path", default=_DEFAULT_LMDB)
    p.add_argument("--test_lmdb_path", default=_DEFAULT_TEST_LMDB)
    p.add_argument("--test_y_lmdb_path", default=_DEFAULT_TEST_Y_LMDB)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap the train+val pool to this many samples (useful for quick runs)")
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--log_dir", default=os.path.join(repo_root, "esdc", "lightning_logs", "pretraining"))
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
        help="Total LR scheduler steps. Defaults to max_epochs * steps_per_epoch, computed from dataset size "
             "and divided by num_gpus for DDP correctness.",
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
        "--y_lmdb_path",
        type=str,
        default=None,
        help="Path to the explicit target LMDB (dataset_y.lmdb). When provided, uses GapFillingTrainingDataModule "
             "instead of LMDBConsecutiveDataModule, enabling direct gap-filling training.",
    )
    p.add_argument(
        "--max_gap_ratio",
        type=float,
        default=None,
        help="Post-hoc filter: drop pre-training samples whose middle-frame primary-variable NaN ratio exceeds "
             "this threshold. Only supported for the plain pre-training path (no --y_lmdb_path).",
    )
    p.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint to fine-tune from. Loads model weights and resets the optimizer "
             "(finetune, not resume). Architecture must match the current config.",
    )
    p.add_argument(
        "--in_channels",
        type=int,
        default=None,
        help="Override the number of input channels derived from meta.json. "
             "Required when training on a dataset with a different aux_vars configuration.",
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


def main():
    args = parse_args()

    if not os.path.exists(args.lmdb_path):
        raise FileNotFoundError(args.lmdb_path)

    _check_lmdb(args.lmdb_path)

    has_test_lmdb = os.path.exists(args.test_lmdb_path)
    if has_test_lmdb:
        _check_lmdb(args.test_lmdb_path)

    has_test_y_lmdb = os.path.exists(args.test_y_lmdb_path)
    if has_test_y_lmdb:
        _check_lmdb(args.test_y_lmdb_path)

    if args.max_gap_ratio is not None and args.y_lmdb_path is not None:
        raise ValueError("--max_gap_ratio is only supported for the plain pre-training path (no --y_lmdb_path).")

    if args.y_lmdb_path is not None:
        if not os.path.exists(args.y_lmdb_path):
            raise FileNotFoundError(args.y_lmdb_path)
        datamodule = GapFillingTrainingDataModule(
            x_lmdb_path=args.lmdb_path,
            y_lmdb_path=args.y_lmdb_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            target_channels=TARGET_CHANNELS,
            max_samples=args.max_samples,
            test_x_lmdb_path=args.test_lmdb_path if (has_test_lmdb and has_test_y_lmdb) else None,
            test_y_lmdb_path=args.test_y_lmdb_path if (has_test_lmdb and has_test_y_lmdb) else None,
        )
    else:
        datamodule = LMDBConsecutiveDataModule(
            lmdb_path=args.lmdb_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            target_channels=TARGET_CHANNELS,
            test_lmdb_path=args.test_lmdb_path if has_test_lmdb else None,
            max_samples=args.max_samples,
            max_gap_ratio=args.max_gap_ratio,
            gap_ratio_channel=TARGET_CHANNELS[0],
        )

    n_total = None
    if args.total_steps is None:
        if args.max_gap_ratio is not None:
            # Raw LMDB entry count doesn't know about the gap_ratio filter applied
            # inside the datamodule -- read the actual post-filter size instead, so
            # the cosine LR schedule's total_steps matches the real number of steps.
            datamodule.setup(stage="fit")
            n_total = len(datamodule.train_ds) + len(datamodule.val_ds)
        else:
            env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
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

    if total_steps < args.warmup_steps:
        print(
            f"WARNING: total_steps ({total_steps}) < warmup_steps ({args.warmup_steps}). "
            "The cosine phase will have T_max=0. Reduce --warmup_steps or increase training length."
        )

    _meta_path = os.path.join(_esdc_data, 'meta.json')
    _norm_range = None
    if os.path.exists(_meta_path):
        with open(_meta_path) as _f:
            _meta = json.load(_f)
        _norm_range = _meta.get('valid_range') or _meta.get('computed_range')
        image_size = _meta['block_size'][1]
        num_frames = _meta['block_size'][0]
        in_channels = args.in_channels if args.in_channels is not None else (1 + len(_meta.get('aux_vars', [])))
    else:
        raise FileNotFoundError(
            f'meta.json not found at {_meta_path}. '
            'Run create_esdc_dataset.py first.'
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
            image_size=image_size,
            patch_size=args.patch_size,
            num_frames=num_frames,
            in_channels=in_channels,
            out_channels=len(TARGET_CHANNELS),
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            decoder_heads=args.decoder_heads,
            lr=args.lr,
            loss_type=args.loss_type,
            norm_range=tuple(_norm_range) if _norm_range is not None else None,
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
            filename=f"lst-gapfiller-{{epoch:02d}}-{{{monitor_key}:.4f}}",
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
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=datamodule)
    should_test = (
        (args.y_lmdb_path is None and has_test_lmdb)
        or (args.y_lmdb_path is not None and has_test_lmdb and has_test_y_lmdb)
    )
    if should_test:
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
