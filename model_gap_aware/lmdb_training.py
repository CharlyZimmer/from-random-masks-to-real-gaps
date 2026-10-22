"""Lightning DataModule and LMDB Dataset utilities for gap-aware training.

LMDB value format (raw):
  int32 ndim, followed by `ndim` int32 shape values, followed by raw float32 data bytes.

Pickle format (legacy) is also supported for backwards compatibility:
  pickle.dumps((array_bytes, shape))

Keys are sequential integer strings: b"0", b"1", ..., b"N-1".

`__getitem__` returns:
  x: (T, C, H, W)
  y: (C_out, H, W) -- the middle frame, selecting `target_channels`.

The model's loss ignores NaNs in y.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

from model_gap_aware.maskfeat import MaskFeatGapFiller


def _parse_int_list(csv: str) -> List[int]:
    csv = csv.strip()
    if not csv:
        return []
    return [int(x) for x in csv.split(",")]


class ConsecutiveLMDBDataset(Dataset):
    """Read consecutive samples from an LMDB with sequential integer-string keys."""

    def __init__(
        self,
        lmdb_path: str,
        target_channels: Sequence[int] = (0,),
        *,
        subdir: bool = False,
        max_gap_ratio: Optional[float] = None,
        gap_ratio_channel: int = 0,
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.target_channels = list(target_channels)
        self.subdir = subdir

        self._env: Optional[lmdb.Environment] = None
        self._txn: Optional[lmdb.Transaction] = None
        self._length = self._infer_length()

        self._index_map: Optional[List[int]] = None
        if max_gap_ratio is not None:
            self._index_map = self._build_gap_filter(max_gap_ratio, gap_ratio_channel)

    def _build_gap_filter(self, max_gap_ratio: float, channel: int) -> List[int]:
        """Post-hoc filter: keep samples whose middle-frame NaN ratio on `channel`
        is <= max_gap_ratio. Single sequential pass over the LMDB at construction
        time -- only exact for datasets without synthetic gap injection (e.g. the
        ESDC pre-training LMDB), where the middle frame's NaN pattern is the raw,
        untouched target.
        """
        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
            subdir=self.subdir,
        )
        kept: List[int] = []
        with env.begin(write=False) as txn:
            for i in range(self._length):
                value = txn.get(str(i).encode())
                arr, _ = self._decode_value(value)
                mid = arr.shape[0] // 2
                ratio = float(np.isnan(arr[mid, channel]).mean())
                if ratio <= max_gap_ratio:
                    kept.append(i)
        env.close()
        print(
            f"ConsecutiveLMDBDataset: gap_ratio filter (<= {max_gap_ratio}) kept "
            f"{len(kept)}/{self._length} samples from {self.lmdb_path}"
        )
        return kept

    def _infer_length(self) -> int:
        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
            subdir=self.subdir,
        )
        with env.begin(write=False) as txn:
            n = int(txn.stat().get("entries", 0))
        env.close()
        return n

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=512,
                subdir=self.subdir,
            )
        return self._env

    def _get_txn(self) -> lmdb.Transaction:
        """Keep a read-only transaction per worker for faster repeated reads."""
        if self._txn is None:
            self._txn = self._get_env().begin(write=False)
        return self._txn

    def __getstate__(self):
        """Make Dataset picklable for worker processes.

        Workers should recreate env/txn after fork/spawn.
        """
        state = dict(self.__dict__)
        state["_env"] = None
        state["_txn"] = None
        return state

    def __len__(self) -> int:
        return len(self._index_map) if self._index_map is not None else self._length

    def _decode_value(self, value: bytes) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Decode LMDB value into a float32 numpy array.

        Supports:
          - pickle format: pickle.dumps((bytes, shape))
          - raw format: int32 ndim + shape int32s + raw float32 bytes
        """
        # Heuristic: raw format starts with small ndim (3 or 4) as int32.
        # If that doesn't look right, fall back to pickle.
        if len(value) >= 4:
            ndim = int(np.frombuffer(value[:4], dtype=np.int32, count=1)[0])
            if 1 <= ndim <= 8 and len(value) >= 4 * (1 + ndim):
                shape = tuple(np.frombuffer(value[4 : 4 * (1 + ndim)], dtype=np.int32, count=ndim).tolist())
                data = value[4 * (1 + ndim) :]
                # Validate byte length (may not match if this was actually pickled)
                if len(data) == int(np.prod(shape)) * 4:
                    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
                    return arr, shape

        # Pickle fallback
        arr_bytes, arr_shape = pickle.loads(value)
        arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(tuple(arr_shape))
        return arr, tuple(arr_shape)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._index_map is not None:
            idx = self._index_map[idx]
        key = str(idx).encode()
        txn = self._get_txn()
        value = txn.get(key)
        if value is None:
            raise KeyError(f"LMDB key not found: {key!r}")

        arr, _ = self._decode_value(value)

        x = torch.from_numpy(arr)  # (T, C, H, W)

        mid = x.shape[0] // 2
        y = x[mid, self.target_channels, :, :]
        return x, y


class LMDBConsecutiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        lmdb_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.05,
        target_channels: Sequence[int] = (0,),
        seed: int = 42,
        prefetch_factor: int = 2,
        *,
        subdir: bool = False,
        pin_memory: bool = True,
        test_lmdb_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        max_gap_ratio: Optional[float] = None,
        gap_ratio_channel: int = 0,
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.target_channels = list(target_channels)
        self.seed = seed
        self.subdir = subdir
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.test_lmdb_path = test_lmdb_path
        self.max_samples = max_samples
        self.max_gap_ratio = max_gap_ratio
        self.gap_ratio_channel = gap_ratio_channel

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.train_ds is None:
            full = ConsecutiveLMDBDataset(
                self.lmdb_path,
                target_channels=self.target_channels,
                subdir=self.subdir,
                max_gap_ratio=self.max_gap_ratio,
                gap_ratio_channel=self.gap_ratio_channel,
            )
            n_total = len(full)
            if self.max_samples is not None and self.max_samples < n_total:
                discard = n_total - self.max_samples
                full, _ = random_split(
                    full,
                    [self.max_samples, discard],
                    generator=torch.Generator().manual_seed(self.seed),
                )
                n_total = self.max_samples
            n_val = max(1, int(self.val_split * n_total))
            n_train = n_total - n_val
            self.train_ds, self.val_ds = random_split(
                full,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage in (None, "test"):
            if self.test_lmdb_path is not None:
                self.test_ds = ConsecutiveLMDBDataset(
                    self.test_lmdb_path,
                    target_channels=self.target_channels,
                    subdir=self.subdir,
                    max_gap_ratio=self.max_gap_ratio,
                    gap_ratio_channel=self.gap_ratio_channel,
                )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "test_lmdb_path was not provided to LMDBConsecutiveDataModule"
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )


class RawLMDBDataset(Dataset):
    """Read tensors from an LMDB as-is, without mid-frame extraction."""

    def __init__(self, lmdb_path: str) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self._env: Optional[lmdb.Environment] = None
        self._txn: Optional[lmdb.Transaction] = None
        self._length = self._infer_length()

    def _infer_length(self) -> int:
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=512, subdir=False)
        with env.begin(write=False) as txn:
            n = int(txn.stat().get("entries", 0))
        env.close()
        return n

    def _get_txn(self) -> lmdb.Transaction:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=512, subdir=False
            )
        if self._txn is None:
            self._txn = self._env.begin(write=False)
        return self._txn

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_env"] = None
        state["_txn"] = None
        return state

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        key = str(idx).encode()
        value = self._get_txn().get(key)
        if value is None:
            raise KeyError(f"LMDB key not found: {key!r}")
        ndim = int(np.frombuffer(value[:4], dtype=np.int32, count=1)[0])
        shape = tuple(np.frombuffer(value[4 : 4 * (1 + ndim)], dtype=np.int32, count=ndim).tolist())
        arr = np.frombuffer(value[4 * (1 + ndim) :], dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy())


class GapFillingTrainingDataset(Dataset):
    """Pairs a ConsecutiveLMDBDataset (input x) with a RawLMDBDataset (explicit target y)."""

    def __init__(
        self,
        x_lmdb_path: str,
        y_lmdb_path: str,
        target_channels: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        self.x_ds = ConsecutiveLMDBDataset(x_lmdb_path, target_channels=target_channels)
        self.y_ds = RawLMDBDataset(y_lmdb_path)
        if len(self.x_ds) != len(self.y_ds):
            raise ValueError(
                f"x LMDB ({len(self.x_ds)} entries) and y LMDB ({len(self.y_ds)} entries) must have the same length"
            )

    def __len__(self) -> int:
        return len(self.x_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.x_ds[idx]
        y = self.y_ds[idx]
        return x, y


class GapFillingTrainingDataModule(pl.LightningDataModule):
    """DataModule for direct gap-filling training using paired x/y LMDBs."""

    def __init__(
        self,
        x_lmdb_path: str,
        y_lmdb_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.05,
        target_channels: Sequence[int] = (0,),
        seed: int = 42,
        prefetch_factor: int = 2,
        *,
        pin_memory: bool = True,
        max_samples: Optional[int] = None,
        test_x_lmdb_path: Optional[str] = None,
        test_y_lmdb_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.x_lmdb_path = x_lmdb_path
        self.y_lmdb_path = y_lmdb_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.target_channels = list(target_channels)
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.max_samples = max_samples
        self.test_x_lmdb_path = test_x_lmdb_path
        self.test_y_lmdb_path = test_y_lmdb_path

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full = GapFillingTrainingDataset(
                self.x_lmdb_path, self.y_lmdb_path, target_channels=self.target_channels
            )
            n_total = len(full)
            if self.max_samples is not None and self.max_samples < n_total:
                discard = n_total - self.max_samples
                full, _ = random_split(
                    full, [self.max_samples, discard], generator=torch.Generator().manual_seed(self.seed)
                )
                n_total = self.max_samples
            n_val = max(1, int(self.val_split * n_total))
            n_train = n_total - n_val
            self.train_ds, self.val_ds = random_split(
                full, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed)
            )
        if stage in (None, "test"):
            if self.test_x_lmdb_path is not None and self.test_y_lmdb_path is not None:
                self.test_ds = GapFillingTrainingDataset(
                    self.test_x_lmdb_path, self.test_y_lmdb_path, target_channels=self.target_channels
                )

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, (
            "test_x_lmdb_path/test_y_lmdb_path were not provided to GapFillingTrainingDataModule"
        )
        return self._loader(self.test_ds, shuffle=False)


class IdxManifest:
    """Loads the `<lmdb_path>.manifest.json` sidecar written by
    create_LMDB.py / create_gap_mask_lmdb.py.

    `create_LMDB.py`'s optional --test_split shuffles the location list
    before sequential LMDB keys are assigned, so the raw idx (SSL-consecutive
    folder name) corresponding to a given LMDB key can't be recovered from
    the LMDB alone. This manifest is a sidecar record of that mapping --
    it does not change the LMDB's own key/value format.
    """

    def __init__(self, lmdb_path: str) -> None:
        manifest_path = lmdb_path + ".manifest.json"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Rebuild this LMDB with the "
                "current create_LMDB.py / create_gap_mask_lmdb.py to generate it."
            )
        with open(manifest_path) as f:
            data = json.load(f)
        self.indices: List[int] = data["indices"]  # indices[key] == idx
        self.idx_to_key: Dict[int, int] = {idx: key for key, idx in enumerate(self.indices)}


class PretrainGapMaskJoinDataset(Dataset):
    """Reconstructs gap-filling training samples by joining a lean gap-mask
    LMDB with one or more pretraining LMDBs, by raw idx (via IdxManifest).

    The pretraining LMDB(s) and the gap-mask LMDB each independently shuffle
    their own location lists before assigning sequential integer keys, so
    samples cannot be paired by position (unlike GapFillingTrainingDataset,
    which is positional and remains correct for ESDC's pre-aligned x/y
    LMDBs -- do not repurpose it for this join). A gap-filling-training
    sample's idx may fall in either a pretraining train LMDB or its test
    LMDB (the two datasets' splits are independent), so every path in
    `pretrain_lmdb_paths` is searched in order.

    Reconstruction, given the pretraining middle frame `pretrain_mid` and the
    donor gap mask `donor_mask` (H, W) bool:
      x_gappy = pretrain_mid.clone(); x_gappy[:, donor_mask] = NaN
      y       = pretrain_mid[target_channels].clone(); y[:, ~donor_mask] = NaN
    Because pretraining GeoTIFFs already store NaN (not 0) at natural
    cloud/boundary pixels, this also correctly carries forward any
    pre-existing natural gaps inside the synthetic gap region into `y` as
    NaN, with no extra bookkeeping -- the model's loss already ignores NaN
    in `y`.

    __getitem__ returns (x_gappy, y):
      x_gappy: (T, C, H, W) float32 torch tensor.
      y:       (C_out, H, W) float32 torch tensor.
    """

    def __init__(
        self,
        pretrain_lmdb_paths: Sequence[str],
        gap_mask_lmdb_path: str,
        target_channels: Sequence[int] = tuple(range(10)),
        *,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.target_channels = list(target_channels)
        self.pretrain_paths = list(pretrain_lmdb_paths)
        self._pretrain_datasets = [
            ConsecutiveLMDBDataset(p, target_channels=self.target_channels)
            for p in self.pretrain_paths
        ]
        self._gap_mask_ds = RawLMDBDataset(gap_mask_lmdb_path)

        pretrain_manifests = [IdxManifest(p) for p in self.pretrain_paths]
        gap_mask_manifest = IdxManifest(gap_mask_lmdb_path)

        # entries[i] = (gap_mask_key, pretrain_src_index, pretrain_key)
        self.entries: List[Tuple[int, int, int]] = []
        missing: List[int] = []
        for gm_key, idx in enumerate(gap_mask_manifest.indices):
            found = None
            for src_i, manifest in enumerate(pretrain_manifests):
                if idx in manifest.idx_to_key:
                    found = (src_i, manifest.idx_to_key[idx])
                    break
            if found is None:
                missing.append(idx)
                continue
            self.entries.append((gm_key, found[0], found[1]))

        if missing:
            msg = (
                f"{len(missing)} gap-mask sample(s) have no matching idx in any of the "
                f"given pretraining LMDB(s) (first few: {missing[:5]}); dropped."
            )
            if strict:
                raise ValueError(msg)
            print(f"WARNING: {msg}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gm_key, src_i, pretrain_key = self.entries[i]
        x_full, _ = self._pretrain_datasets[src_i][pretrain_key]  # (T, C, H, W)
        gap_mask_raw = self._gap_mask_ds[gm_key]  # (1, H, W) float32 in {0, 1}
        donor_mask = gap_mask_raw[0] > 0.5  # (H, W) bool

        mid = x_full.shape[0] // 2
        x_gappy = x_full.clone()
        x_gappy[mid, :, donor_mask] = float("nan")

        y = x_full[mid, self.target_channels, :, :].clone()
        y[:, ~donor_mask] = float("nan")
        return x_gappy, y


class PretrainGapMaskJoinDataModule(pl.LightningDataModule):
    """DataModule for direct gap-filling training via PretrainGapMaskJoinDataset."""

    def __init__(
        self,
        pretrain_lmdb_paths: Sequence[str],
        gap_mask_lmdb_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.05,
        target_channels: Sequence[int] = (0,),
        seed: int = 42,
        prefetch_factor: int = 2,
        *,
        pin_memory: bool = True,
        max_samples: Optional[int] = None,
        test_gap_mask_lmdb_path: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.pretrain_lmdb_paths = list(pretrain_lmdb_paths)
        self.gap_mask_lmdb_path = gap_mask_lmdb_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.target_channels = list(target_channels)
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.max_samples = max_samples
        self.test_gap_mask_lmdb_path = test_gap_mask_lmdb_path
        self.strict = strict

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full = PretrainGapMaskJoinDataset(
                self.pretrain_lmdb_paths,
                self.gap_mask_lmdb_path,
                target_channels=self.target_channels,
                strict=self.strict,
            )
            n_total = len(full)
            if self.max_samples is not None and self.max_samples < n_total:
                discard = n_total - self.max_samples
                full, _ = random_split(
                    full, [self.max_samples, discard], generator=torch.Generator().manual_seed(self.seed)
                )
                n_total = self.max_samples
            n_val = max(1, int(self.val_split * n_total))
            n_train = n_total - n_val
            self.train_ds, self.val_ds = random_split(
                full, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed)
            )
        if stage in (None, "test"):
            if self.test_gap_mask_lmdb_path is not None:
                self.test_ds = PretrainGapMaskJoinDataset(
                    self.pretrain_lmdb_paths,
                    self.test_gap_mask_lmdb_path,
                    target_channels=self.target_channels,
                    strict=self.strict,
                )

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, (
            "test_gap_mask_lmdb_path was not provided to PretrainGapMaskJoinDataModule"
        )
        return self._loader(self.test_ds, shuffle=False)


def build_model(
    image_size: int,
    patch_size: int,
    num_frames: int,
    in_channels: int = 1,
    out_channels: int = 1,
    embed_dim: int = 192,
    depth: int = 1,
    num_heads: int = 1,
    decoder_heads: int = 1,
    lr: float = 1e-4,
    loss_type: str = "mae",
    norm_range: Optional[Tuple[float, float]] = None,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    gap_only_masking: bool = False,
    structured_masking: bool = False,
) -> MaskFeatGapFiller:
    model = MaskFeatGapFiller(
        image_size=image_size,
        patch_size=patch_size,
        num_frames=num_frames,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        depth=depth,
        heads=num_heads,
        decoder_heads=decoder_heads,
        lr=lr,
        loss_type=loss_type,
        norm_range=norm_range,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        gap_only_masking=gap_only_masking,
        structured_masking=structured_masking,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskFeatGapFiller on LMDB consecutive dataset")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)

    # Data / model shape
    parser.add_argument("--image_size", type=int, default=128, help="H=W of frames in pixels")
    parser.add_argument("--num_frames", type=int, default=5, help="T (time steps)")
    parser.add_argument("--in_channels", type=int, default=3, help="C (bands used as input)")
    parser.add_argument("--vit_patch_size", type=int, default=8, help="Transformer patch size")

    parser.add_argument(
        "--target_channels",
        type=str,
        default="0",
        help="Comma-separated channel indices in the middle frame to reconstruct (e.g. '0,1,2').",
    )
    parser.add_argument("--log_dir", type=str, default="./lightning_logs/pretraining")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--subdir", action="store_true", help="Set if LMDB is a directory (subdir=True)")
    parser.add_argument("--loss_type", type=str, default="mae", choices=["mae", "mse"])

    args = parser.parse_args()

    target_channels = _parse_int_list(args.target_channels)
    if len(target_channels) == 0:
        raise ValueError("--target_channels must contain at least one channel index")

    if not os.path.exists(args.lmdb_path):
        raise FileNotFoundError(f"LMDB file not found: {args.lmdb_path}")

    datamodule = LMDBConsecutiveDataModule(
        lmdb_path=args.lmdb_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        target_channels=target_channels,
        subdir=args.subdir,
    )

    model = build_model(
        image_size=args.image_size,
        patch_size=args.vit_patch_size,
        num_frames=args.num_frames,
        in_channels=args.in_channels,
        out_channels=len(target_channels),
        loss_type=args.loss_type,
    )

    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

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

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
