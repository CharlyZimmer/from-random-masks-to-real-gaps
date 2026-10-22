import lmdb

import numpy as np
from pathlib import Path
import os
import h5py
from tqdm import tqdm
import torch

# -------- CONFIG --------
dir_path = os.environ.get(
    "ESDC_DATA_DIR",
    "./esdc/data/lst_no_syn_gaps",
)
if not dir_path.endswith("/"):
    dir_path += "/"
HDF5_PATH = dir_path + "dataset.h5"
OUT_DIR = dir_path + "lmdb/"
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Remove existing LMDB files so old keys don't persist alongside new ones
for fname in ("dataset.lmdb", "dataset.lmdb-lock"):
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)

env = lmdb.open(
    OUT_DIR + "dataset.lmdb",
    map_size=150 * 1024**3,
    subdir=False,
)

def _write_lmdb(env, X):
    """Write X (N, T, H, W, C) channels-last array to an open LMDB env (raw format)."""
    with env.begin(write=True) as txn:
        for i in tqdm(range(len(X))):
            x_np = np.ascontiguousarray(
                np.transpose(X[i], (0, 3, 1, 2)).astype(np.float32)
            )  # (T, C, H, W)
            shape = np.asarray(x_np.shape, dtype=np.int32)
            header = np.concatenate([np.asarray([shape.size], dtype=np.int32), shape]).tobytes()
            txn.put(str(i).encode(), header + x_np.tobytes())


with h5py.File(HDF5_PATH, "r") as f:
    print("Writing training LMDB …")
    _write_lmdb(env, f['X_train'])

    # Test LMDB — same raw format, sequential keys 0..N_test-1
    for fname in ("dataset_test.lmdb", "dataset_test.lmdb-lock"):
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    env_test = lmdb.open(
        OUT_DIR + "dataset_test.lmdb",
        map_size=150 * 1024**3,
        subdir=False,
    )
    print("Writing test LMDB …")
    _write_lmdb(env_test, f['X_test'])
    env_test.close()




