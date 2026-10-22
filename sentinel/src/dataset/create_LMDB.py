import os
import argparse
import json
import math
from typing import List, Tuple, Optional, Sequence
from collections.abc import Sized
import shutil

import lmdb
import numpy as np
import pickle
import rasterio
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _sentinel2_normalize(
    x: np.ndarray,
    *,
    scale_factor: float = 10000.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Common Sentinel-2 normalization.

    Typical L2A/L1C products are stored as scaled integers where
    reflectance ~= DN / 10000.

    Steps:
      1) scale to reflectance: x /= scale_factor
      2) clip into [clip_min, clip_max]
      3) optional per-band standardization: (x - mean) / std

    Expects x shaped (T, C, H, W) or (C, H, W).

    Note: This function preserves NaNs (used as cloud/gap markers). We therefore
    only clip finite values.
    """
    x = x.astype(np.float32, copy=False)
    x = x / float(scale_factor)

    # Preserve NaNs: np.clip would turn NaNs into clip_max.
    finite = np.isfinite(x)
    x[finite] = np.clip(x[finite], clip_min, clip_max)

    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("Both mean and std must be provided together for standardization")

        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        if mean_arr.ndim != 1 or std_arr.ndim != 1 or mean_arr.shape != std_arr.shape:
            raise ValueError("mean and std must be 1D arrays of the same length")

        # Broadcast over spatial and (optional) time dims.
        # For (T,C,H,W): reshape to (1,C,1,1). For (C,H,W): reshape to (C,1,1).
        if x.ndim == 4:
            mean_b = mean_arr.reshape(1, -1, 1, 1)
            std_b = std_arr.reshape(1, -1, 1, 1)
        elif x.ndim == 3:
            mean_b = mean_arr.reshape(-1, 1, 1)
            std_b = std_arr.reshape(-1, 1, 1)
        else:
            raise ValueError(f"Unexpected input ndim={x.ndim}; expected 3 or 4")

        if mean_arr.shape[0] != x.shape[-3]:
            raise ValueError(
                f"mean/std length ({mean_arr.shape[0]}) must match channel dimension ({x.shape[-3]})"
            )

        x = (x - mean_b) / (std_b + 1e-6)

    return x


class S2ConsecutiveDataset(Dataset):
    """Dataset over data/SSL-consecutive.

    Expects structure:
        root/SSL-consecutive/{idx:06d}/{timestamp}/S2C.tif, metadata.json

    S2C.tif is a multi-band GeoTIFF (one file per timestamp) with bands in
    spectral order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.

    Each __getitem__ returns a time series stacked as (T, C, H, W), where
    T = number of time steps (inferred from subfolders) and C = number of bands.

    Optionally applies Sentinel-2 reflectance normalization and clipping.
    """

    def __init__(
        self,
        root: str,
        subdir: str = "SSL-consecutive",
        *,
        locations: Optional[List[str]] = None,
        normalize: bool = False,
        scale_factor: float = 10000.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        self.root = root
        self.subdir = subdir
        self.base_dir = os.path.join(root, subdir)
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mean = mean
        self.std = std

        if locations is not None:
            self.locations = list(locations)
        else:
            # One folder per location: 000000, 000001, ...
            self.locations = sorted(
                d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))
            )

    def __len__(self) -> int:
        return len(self.locations)

    def _read_patch(self, t_dir: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Read multi-band S2C.tif from a timestamp directory and return (C, H, W)."""
        path = os.path.join(t_dir, "S2C.tif")
        with rasterio.open(path) as src:
            stacked = src.read().astype(np.float32)  # (C, H, W)
        height, width = stacked.shape[1], stacked.shape[2]

        # Pixels where all bands are 0 were cloud-masked by the downloader
        # (sampleRectangle defaultValue=0). Convert to NaN so torch.isnan()
        # gap detection in the model works correctly.
        gap_mask = np.all(stacked == 0, axis=0)  # (H, W)
        stacked[:, gap_mask] = np.nan

        if self.normalize:
            stacked = _sentinel2_normalize(
                stacked,
                scale_factor=self.scale_factor,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                mean=self.mean,
                std=self.std,
            )

        return stacked, (height, width)

    def __getitem__(self, idx: int):
        loc_id = self.locations[idx]
        loc_dir = os.path.join(self.base_dir, loc_id)
        # subfolders are time steps
        time_steps = sorted(
            d for d in os.listdir(loc_dir)
            if os.path.isdir(os.path.join(loc_dir, d))
        )
        patches = []
        for t in time_steps:
            t_dir = os.path.join(loc_dir, t)
            patch, _ = self._read_patch(t_dir)
            patches.append(patch[np.newaxis, ...])  # (1, C, H, W)
        if not patches:
            raise RuntimeError(f"No time steps found in {loc_dir}")
        # (T, C, H, W)
        s2c = np.concatenate(patches, axis=0).astype(np.float32)

        # If normalization is enabled, we already normalized per timestamp. No extra step needed.
        return s2c


def _remove_lmdb_path(lmdb_path: str) -> None:
    """Remove an LMDB path if it already exists.

    LMDB may be created either as a directory or a single file depending on platform/config.
    Also removes the associated '-lock' file if present.
    """
    lmdb_path = os.path.abspath(lmdb_path)
    lock_path = lmdb_path + "-lock"

    if os.path.isdir(lmdb_path):
        shutil.rmtree(lmdb_path)
    elif os.path.exists(lmdb_path):
        os.remove(lmdb_path)

    if os.path.exists(lock_path):
        os.remove(lock_path)


def _write_idx_manifest(
    lmdb_file: str, locations: List[str], *, root: str, subdir: str
) -> None:
    """Persist the exact idx-order used to assign sequential LMDB keys.

    `create_LMDB.py`'s optional --test_split shuffles the location list before
    keys are assigned, so nothing else records which raw idx (folder name)
    corresponds to a given LMDB key. This sidecar manifest (not a change to
    the LMDB's own key/value format) lets other code join by idx instead of
    position -- e.g. sentinel_gapfilling_training_downloader.py's gap-mask
    LMDB reuses this pretraining LMDB's imagery by idx.
    """
    indices = [int(loc) for loc in locations]
    manifest = {
        "version": 1,
        "lmdb_file": os.path.basename(lmdb_file),
        "source_root": os.path.abspath(root),
        "subdir": subdir,
        "count": len(indices),
        "indices": indices,
    }
    with open(lmdb_file + ".manifest.json", "w") as f:
        json.dump(manifest, f)


def make_lmdb_consecutive(
    dataset: Dataset,
    lmdb_file: str,
    num_workers: int = 6,
    map_size: int = 1099511627776,  # 1 TB
    show_progress: bool = True,
    overwrite: bool = True,
    *,
    value_format: str = "pickle",
) -> None:
    """Create an LMDB from the consecutive Sentinel-2 dataset.

    Each entry key is `str(index).encode()`.

    value_format:
      - "pickle" (default): pickle.dumps((bytes, shape))
      - "raw": store a small binary header + raw float32 bytes (faster to decode)

    For "raw": value layout is:
      int32 ndim, followed by `ndim` int32 shape values, followed by float32 data bytes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(lmdb_file)), exist_ok=True)

    if overwrite:
        _remove_lmdb_path(lmdb_file)

    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],  # return the single item directly
    )

    # These options can improve read performance later.
    env = lmdb.open(
        lmdb_file,
        map_size=map_size,
        writemap=True,
        map_async=True,
        metasync=False,
        sync=False,
        subdir=False,
    )
    txn = env.begin(write=True)

    iterator = enumerate(loader)
    if show_progress:
        if tqdm is None:
            raise ImportError(
                "tqdm is required for progress bars. Install it with: pip install tqdm, or pass --no_progress"
            )
        iterator = tqdm(iterator, total=len(dataset) if isinstance(dataset, Sized) else None, desc="Writing LMDB", unit="sample")

    for index, s2c in iterator:
        s2c_np = np.asarray(s2c, dtype=np.float32)  # (T, C, H, W)

        if value_format == "pickle":
            obj = (s2c_np.tobytes(), s2c_np.shape)
            value = pickle.dumps(obj)
        elif value_format == "raw":
            shape = np.asarray(s2c_np.shape, dtype=np.int32)
            header = np.concatenate([np.asarray([shape.size], dtype=np.int32), shape]).tobytes()
            value = header + s2c_np.tobytes()
        else:
            raise ValueError(f"Unknown value_format={value_format!r}; expected 'pickle' or 'raw'")

        txn.put(str(index).encode(), value)

        if index % 1000 == 0 and index > 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.sync()
    env.close()


def _parse_float_list(csv: str) -> Sequence[float]:
    csv = csv.strip()
    if not csv:
        return []
    return [float(x) for x in csv.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LMDB from SSL-consecutive patches")
    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="Root directory containing SSL-consecutive",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="SSL-consecutive",
        help="Subfolder with consecutive patches",
    )
    parser.add_argument(
        "--lmdb_file",
        type=str,
        default="./data/ssl4eo_consecutive_s2c.lmdb",
        help="Output LMDB path",
    )

    # Normalization options
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply common Sentinel-2 normalization: divide by 10000 and clip to [0,1] (optional mean/std).",
    )
    parser.add_argument("--scale_factor", type=float, default=10000.0, help="Scale factor for DN->reflectance")
    parser.add_argument("--clip_min", type=float, default=0.0, help="Minimum reflectance after scaling")
    parser.add_argument("--clip_max", type=float, default=1.0, help="Maximum reflectance after scaling")
    parser.add_argument(
        "--mean",
        type=str,
        default="",
        help="Optional comma-separated per-band mean for standardization (must match #bands).",
    )
    parser.add_argument(
        "--std",
        type=str,
        default="",
        help="Optional comma-separated per-band std for standardization (must match #bands).",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of DataLoader workers for LMDB creation",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the progress bar",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not delete an existing LMDB before writing",
    )
    parser.add_argument(
        "--value_format",
        type=str,
        default="pickle",
        choices=["pickle", "raw"],
        help="How to encode LMDB values. 'raw' is faster to decode during training.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.0,
        help="Fraction of locations to hold out as a test set (0 = no split).",
    )
    parser.add_argument(
        "--test_lmdb_file",
        type=str,
        default="",
        help="Output path for the test LMDB. Auto-derived from --lmdb_file if empty.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train/test location shuffle.",
    )
    args = parser.parse_args()

    mean = _parse_float_list(args.mean)
    std = _parse_float_list(args.std)
    if (len(mean) > 0) != (len(std) > 0):
        raise ValueError("--mean and --std must be provided together")

    norm_kwargs = dict(
        normalize=bool(args.normalize),
        scale_factor=float(args.scale_factor),
        clip_min=float(args.clip_min),
        clip_max=float(args.clip_max),
        mean=mean if len(mean) > 0 else None,
        std=std if len(std) > 0 else None,
    )

    if args.test_split > 0.0:
        base_dir = os.path.join(args.root, args.subdir)
        all_locations = sorted(
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        )
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_locations)
        n_test = max(1, math.ceil(len(all_locations) * args.test_split))
        train_locations = all_locations[:-n_test]
        test_locations = all_locations[-n_test:]

        test_lmdb_file = args.test_lmdb_file or args.lmdb_file.replace(".lmdb", "_test.lmdb")

        train_dataset = S2ConsecutiveDataset(
            root=args.root, subdir=args.subdir, locations=train_locations, **norm_kwargs
        )
        test_dataset = S2ConsecutiveDataset(
            root=args.root, subdir=args.subdir, locations=test_locations, **norm_kwargs
        )

        print(f"Train locations: {len(train_locations)}, test locations: {len(test_locations)}")
        make_lmdb_consecutive(
            train_dataset,
            lmdb_file=args.lmdb_file,
            num_workers=args.num_workers,
            show_progress=not args.no_progress,
            overwrite=not args.no_overwrite,
            value_format=args.value_format,
        )
        _write_idx_manifest(args.lmdb_file, train_dataset.locations, root=args.root, subdir=args.subdir)
        make_lmdb_consecutive(
            test_dataset,
            lmdb_file=test_lmdb_file,
            num_workers=args.num_workers,
            show_progress=not args.no_progress,
            overwrite=not args.no_overwrite,
            value_format=args.value_format,
        )
        _write_idx_manifest(test_lmdb_file, test_dataset.locations, root=args.root, subdir=args.subdir)
    else:
        dataset = S2ConsecutiveDataset(root=args.root, subdir=args.subdir, **norm_kwargs)
        make_lmdb_consecutive(
            dataset,
            lmdb_file=args.lmdb_file,
            num_workers=args.num_workers,
            show_progress=not args.no_progress,
            overwrite=not args.no_overwrite,
            value_format=args.value_format,
        )
        _write_idx_manifest(args.lmdb_file, dataset.locations, root=args.root, subdir=args.subdir)


if __name__ == "__main__":
    main()