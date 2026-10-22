"""Create a lean LMDB of synthetic gap masks from the Sentinel-2 gap-filling
*training* directory.

Reads `${S2_GAP_FILLING_TRAINING_DIR}/gap_masks/{idx:06d}/gap_mask.tif`
(written by sentinel_gapfilling_training_downloader.py) and writes:

  - lmdb/gap_masks.lmdb      : (1, H, W) float32 in {0.0, 1.0}, train split
  - lmdb/gap_masks_test.lmdb : held-out location split

Both LMDBs get a `<file>.manifest.json` sidecar (via create_LMDB.py's
_write_idx_manifest) recording which raw idx corresponds to each sequential
LMDB key -- this is how model_gap_aware.lmdb_training.PretrainGapMaskJoinDataset
joins these masks back to the pretraining LMDB's own (independently shuffled)
keys for the same idx.

No full imagery is read or written here -- the gappy input series and the
ground-truth middle frame are reconstructed at training time by combining a
gap mask from here with the corresponding sample already stored in the
pretraining LMDB.

Run from project root:
    PYTHONPATH=. python sentinel/src/dataset/create_gap_mask_lmdb.py --test_split 0.2
"""
import argparse
import math
import os

import numpy as np
import rasterio
from torch.utils.data import Dataset

from sentinel.src.dataset.create_LMDB import _write_idx_manifest, make_lmdb_consecutive

repo_root = os.environ.get(
    "GAP_AWARE_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", ".."),
)

_DEFAULT_DIR = os.environ.get(
    "S2_GAP_FILLING_TRAINING_DIR",
    os.path.join(repo_root, "sentinel", "data", "gap_filling_training"),
)


class GapMaskDataset(Dataset):
    """Loads root/gap_masks/{idx:06d}/gap_mask.tif -> (1, H, W) float32, values in {0.0, 1.0}."""

    def __init__(self, root: str, locations=None, subdir: str = "gap_masks") -> None:
        self.root = root
        self.subdir = subdir
        self.base_dir = os.path.join(root, subdir)
        if locations is not None:
            self.locations = list(locations)
        else:
            self.locations = sorted(
                d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))
            )

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, idx: int) -> np.ndarray:
        loc_id = self.locations[idx]
        mask_path = os.path.join(self.base_dir, loc_id, "gap_mask.tif")
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # (H, W), values in {0.0, 1.0}
        return mask[np.newaxis, ...]  # (1, H, W)


def _split_locations(base_dir: str, test_split: float, seed: int):
    all_locations = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    if test_split <= 0.0:
        return all_locations, []

    rng = np.random.default_rng(seed)
    rng.shuffle(all_locations)
    n_test = max(1, math.ceil(len(all_locations) * test_split))
    train_locations = all_locations[:-n_test]
    test_locations = all_locations[-n_test:]
    return train_locations, test_locations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a lean gap-mask LMDB from the Sentinel-2 gap-filling training directory"
    )
    parser.add_argument("--root", type=str, default=_DEFAULT_DIR,
                        help="Directory containing gap_masks/")
    parser.add_argument("--out_dir", type=str, default="",
                        help="Output directory for LMDBs. Defaults to <root>/lmdb/")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Fraction of locations to hold out as a test set (0 = no split). "
                             "Independent of the pretraining LMDB's own train/test split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the train/test location shuffle.")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    gm_dir = os.path.join(args.root, "gap_masks")
    if not os.path.isdir(gm_dir):
        raise FileNotFoundError(f"gap_masks/ not found under {args.root}")

    out_dir = args.out_dir or os.path.join(args.root, "lmdb")
    os.makedirs(out_dir, exist_ok=True)

    train_locations, test_locations = _split_locations(gm_dir, args.test_split, args.seed)
    print(f"Train locations: {len(train_locations)}, test locations: {len(test_locations)}")

    def _build(locations, name):
        if not locations:
            return
        dataset = GapMaskDataset(root=args.root, locations=locations)
        lmdb_file = os.path.join(out_dir, name)
        make_lmdb_consecutive(
            dataset,
            lmdb_file=lmdb_file,
            num_workers=args.num_workers,
            show_progress=not args.no_progress,
            overwrite=True,
            value_format="raw",
        )
        _write_idx_manifest(lmdb_file, locations, root=args.root, subdir="gap_masks")

    print("Writing train gap-mask LMDB …")
    _build(train_locations, "gap_masks.lmdb")

    if test_locations:
        print("Writing test gap-mask LMDB …")
        _build(test_locations, "gap_masks_test.lmdb")

    print("Done.")


if __name__ == "__main__":
    main()
