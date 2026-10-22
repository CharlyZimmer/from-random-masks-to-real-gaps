"""Create paired x/y LMDBs from a gap-filling training HDF5.

Reads the HDF5 produced by create_esdc_dataset.py --syn_gaps --no_split and writes:
  - lmdb/dataset.lmdb      : X_train (T, C, H, W) channels-first, one entry per sample
  - lmdb/dataset_y.lmdb   : y_train (1, H, W), valid only at synthetic gap pixels (NaN elsewhere)

The y LMDB stores the LST channel only from y_train, preserving NaN at non-gap pixels so
the training loss (computed at patch_mask=1 & ~isnan(y)) is restricted to gap pixels.
"""

import os

import h5py
import lmdb
import numpy as np
from tqdm import tqdm

dir_path = os.environ.get(
    "ESDC_GAP_FILLING_TRAINING_DIR",
    "./esdc/data/lst_gap_filling_training",
)
if not dir_path.endswith("/"):
    dir_path += "/"

HDF5_PATH = dir_path + "dataset.h5"
OUT_DIR = dir_path + "lmdb/"

os.makedirs(OUT_DIR, exist_ok=True)


def _write_array_lmdb(
    path: str, arrays, n: int, map_size: int, chunk_size: int = 1000
) -> None:
    """Write a sequence of float32 numpy arrays to an LMDB (raw int32 header + float32 bytes)."""
    for fname in (os.path.basename(path), os.path.basename(path) + "-lock"):
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    env = lmdb.open(path, map_size=map_size, subdir=False)
    txn = env.begin(write=True)
    for i, arr in enumerate(tqdm(arrays, total=n)):
        arr = np.ascontiguousarray(arr.astype(np.float32))
        shape = np.asarray(arr.shape, dtype=np.int32)
        header = np.concatenate([np.asarray([shape.size], dtype=np.int32), shape]).tobytes()
        txn.put(str(i).encode(), header + arr.tobytes())
        if (i + 1) % chunk_size == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()


with h5py.File(HDF5_PATH, "r") as f:
    X_train = f["X_train"]  # (N, T, H, W, C)
    y_train = f["y_train"]  # (N, 1, H, W, 1)

    N = X_train.shape[0]
    print(f"Dataset: {N} samples")

    x_sample_bytes = int(np.prod(X_train.shape[1:])) * 4  # float32; same total after transpose
    y_sample_bytes = int(np.prod((1, X_train.shape[2], X_train.shape[3]))) * 4
    x_map_size = max(2 * 1024 ** 3, N * x_sample_bytes * 2)
    y_map_size = max(2 * 1024 ** 3, N * y_sample_bytes * 2)

    print("Writing x LMDB (dataset.lmdb) …")
    x_iter = (np.transpose(X_train[i], (0, 3, 1, 2)) for i in range(N))  # (T, C, H, W)
    _write_array_lmdb(OUT_DIR + "dataset.lmdb", x_iter, n=N, map_size=x_map_size)

    print("Writing y LMDB (dataset_y.lmdb) …")
    # y_train[i] shape: (1, H, W, 1) → take channel 0 → (H, W) → add C dim → (1, H, W)
    y_iter = (y_train[i, 0, :, :, 0:1].transpose(2, 0, 1) for i in range(N))  # (1, H, W)
    _write_array_lmdb(OUT_DIR + "dataset_y.lmdb", y_iter, n=N, map_size=y_map_size)

    # Test split (empty for datasets created with --no_split): skip entirely rather
    # than writing empty LMDB files, so downstream code can tell "no test set" apart
    # from "test set with 0 entries" purely from file existence.
    N_test = f["X_test"].shape[0]
    if N_test > 0:
        X_test = f["X_test"]
        y_test = f["y_test"]
        print(f"Test split: {N_test} samples")

        x_test_map_size = max(2 * 1024 ** 3, N_test * x_sample_bytes * 2)
        y_test_map_size = max(2 * 1024 ** 3, N_test * y_sample_bytes * 2)

        print("Writing x test LMDB (dataset_test.lmdb) …")
        x_test_iter = (np.transpose(X_test[i], (0, 3, 1, 2)) for i in range(N_test))  # (T, C, H, W)
        _write_array_lmdb(OUT_DIR + "dataset_test.lmdb", x_test_iter, n=N_test, map_size=x_test_map_size)

        print("Writing y test LMDB (dataset_test_y.lmdb) …")
        y_test_iter = (y_test[i, 0, :, :, 0:1].transpose(2, 0, 1) for i in range(N_test))  # (1, H, W)
        _write_array_lmdb(OUT_DIR + "dataset_test_y.lmdb", y_test_iter, n=N_test, map_size=y_test_map_size)

print("Done.")
