import argparse
import glob
import os
import re
import sys

repo_root = os.environ.get("GAP_AWARE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model_gap_aware.lmdb_training import LMDBConsecutiveDataModule
from model_gap_aware.maskfeat import MaskFeatGapFiller
import pytorch_lightning as pl


_DEFAULT_LMDB = os.environ.get(
    "S2_LMDB_PATH",
    os.path.join(repo_root, "sentinel", "data", "ssl4eo_consecutive_s2c.lmdb"),
)
_DEFAULT_TEST_LMDB = _DEFAULT_LMDB.replace(".lmdb", "_test.lmdb")

TARGET_CHANNELS = list(range(10))  # all 10 Sentinel-2 bands


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


def parse_args():
    p = argparse.ArgumentParser(description="Compute Sentinel-2 test loss from a saved checkpoint")
    p.add_argument("--test_lmdb_path", default=_DEFAULT_TEST_LMDB)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=8)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--log_dir", default=os.path.join(repo_root, "sentinel", "lightning_logs", "pretraining"))
    p.add_argument("--checkpoint", default=None, help="Explicit checkpoint path; overrides --log_dir auto-discovery")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = find_checkpoint(args.log_dir, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    model = MaskFeatGapFiller.load_from_checkpoint(ckpt_path)

    # lmdb_path is only used for train/val stage; passing test path is safe since we only run test.
    datamodule = LMDBConsecutiveDataModule(
        lmdb_path=args.test_lmdb_path,
        test_lmdb_path=args.test_lmdb_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        target_channels=TARGET_CHANNELS,
        subdir=False,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
