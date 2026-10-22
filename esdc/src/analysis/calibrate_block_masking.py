"""
Calibrate ViViT's `block_size_range` against exp01_zero_shot's actual
random_masking ratio distribution.

Background: `ViViT.block_masking` sizes its rectangle independently of any
pre-existing gap patches, then OR's it with the deterministic gap mask — so
the final middle-frame mask ratio is always >= the block's own target and
typically larger. `ViViT.random_masking` avoids this: it computes the total
masked budget first (`mask_ratio_sample ~ Uniform(mask_ratio, 1.0)`) and fills
it including whatever gap patches already exist "for free", so its actual
ratio distribution is (almost) exactly Uniform(mask_ratio, 1.0).

This script loads real middle-frame gap patterns from an ESDC LMDB, computes
random_masking's actual ratio distribution on them as the empirical target,
then grid-searches `block_size_range` (lower, upper) pairs for block_masking,
scoring each candidate by the Kolmogorov-Smirnov statistic against the
target — averaged over several independent data subsamples for robustness.

Run from project root:
    PYTHONPATH=. python esdc/src/analysis/calibrate_block_masking.py
"""
import argparse
import os
import sys

import lmdb
import numpy as np
import torch
from scipy import stats

repo_root = os.environ.get(
    "GAP_AWARE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model_gap_aware.vivit import ViViT

NUM_FRAMES = 5


def _decode(data: bytes) -> np.ndarray:
    ndim = int(np.frombuffer(data, dtype=np.int32, count=1, offset=0)[0])
    shape = tuple(np.frombuffer(data, dtype=np.int32, count=ndim, offset=4).tolist())
    return np.frombuffer(data, dtype=np.float32, offset=4 * (1 + ndim)).reshape(shape)


def load_middle_frame_gap_masks(lmdb_path, patch_size, image_size, n_samples, seed):
    """Return (N, L) bool: True where a middle-frame patch contains a real (NaN) gap."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    n_total = env.stat()["entries"]
    n_samples = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, n_samples, replace=False)

    out = []
    with env.begin(write=False) as txn:
        for idx in indices:
            data = txn.get(str(int(idx)).encode())
            arr = _decode(data)  # (T, C, H, W)
            frame = arr[arr.shape[0] // 2]
            C, H, W = frame.shape
            n_ph, n_pw = H // patch_size, W // patch_size
            r = frame.reshape(C, n_ph, patch_size, n_pw, patch_size)
            out.append(np.isnan(r).any(axis=(0, 2, 4)).reshape(-1))
    env.close()
    return np.stack(out)


def _mid_frame_ratios(model, forced_real, use_block, seed):
    N, L = forced_real.shape
    torch.manual_seed(seed)
    x = torch.randn(N, NUM_FRAMES, L, model.dim)
    forced_mask = torch.zeros(N, NUM_FRAMES, L, dtype=torch.long)
    forced_mask[:, NUM_FRAMES // 2, :] = torch.from_numpy(forced_real.astype(np.int64))
    masking_fn = model.block_masking if use_block else (
        lambda x, forced_mask, gap_only_masking: model.random_masking(
            x, model.mask_ratio, forced_mask, gap_only_masking=gap_only_masking
        )
    )
    full_mask = masking_fn(x, forced_mask, gap_only_masking=False)
    return full_mask[:, NUM_FRAMES // 2, :].float().mean(dim=1).numpy()


def target_distribution(forced_real, image_size, patch_size, mask_ratio, seed):
    """exp01's actual random_masking ratio distribution on real gap patterns."""
    model = ViViT(image_size=image_size, patch_size=patch_size, num_frames=NUM_FRAMES,
                  dim=8, depth=1, heads=1, in_channels=1, mask_ratio=mask_ratio)
    return _mid_frame_ratios(model, forced_real, use_block=False, seed=seed)


def block_distribution(forced_real, image_size, patch_size, block_size_range, seed):
    """block_masking's ratio distribution on the same real gap patterns."""
    model = ViViT(image_size=image_size, patch_size=patch_size, num_frames=NUM_FRAMES,
                  dim=8, depth=1, heads=1, in_channels=1, structured_masking=True,
                  block_size_range=block_size_range, block_aspect_ratio_range=(0.3, 1 / 0.3))
    return _mid_frame_ratios(model, forced_real, use_block=True, seed=seed)


def _parse_range(csv):
    start, stop, step = (float(v) for v in csv.split(":"))
    return np.round(np.arange(start, stop, step), 2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate ViViT block_size_range against exp01's random_masking ratio distribution"
    )
    p.add_argument(
        "--lmdb_path",
        default=os.path.join(
            os.environ.get("ESDC_DATA_DIR", os.path.join(repo_root, "esdc", "data", "lst_no_syn_gaps")),
            "lmdb", "dataset.lmdb",
        ),
    )
    p.add_argument("--patch_size", type=int, default=4, help="Must match shared_config.yaml's model.patch_size")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--mask_ratio", type=float, default=0.7, help="exp01's mask_ratio (target lower bound)")
    p.add_argument("--n_samples", type=int, default=1500, help="Real LMDB samples per data seed")
    p.add_argument("--n_data_seeds", type=int, default=3, help="Independent data subsamples to average KS over")
    p.add_argument("--lower_bounds", default="0.55:0.70:0.01", help="start:stop:step for candidate lower bounds")
    p.add_argument("--upper_bounds", default="0.85:1.00:0.03", help="start:stop:step for candidate upper bounds")
    p.add_argument("--top_k", type=int, default=15, help="How many ranked candidates to print")
    return p.parse_args()


def main():
    args = parse_args()
    lowers = _parse_range(args.lower_bounds)
    uppers = _parse_range(args.upper_bounds)

    results = {}
    for data_seed in range(args.n_data_seeds):
        forced_real = load_middle_frame_gap_masks(
            args.lmdb_path, args.patch_size, args.image_size, args.n_samples, data_seed
        )
        gap_fracs = forced_real.mean(axis=1)
        print(f"[seed {data_seed}] real gap coverage: mean={gap_fracs.mean():.3f} median={np.median(gap_fracs):.3f}")

        target = target_distribution(forced_real, args.image_size, args.patch_size, args.mask_ratio, seed=data_seed)

        for lo in lowers:
            for hi in uppers:
                if lo >= hi:
                    continue
                ratios = block_distribution(forced_real, args.image_size, args.patch_size, (lo, hi), seed=data_seed)
                ks = stats.ks_2samp(ratios, target).statistic
                results.setdefault((lo, hi), []).append(ks)

    print(f"\n{'(lower, upper)':>18} | {'mean KS':>8}")
    ranked = sorted(results.items(), key=lambda kv: np.mean(kv[1]))
    for (lo, hi), ks_list in ranked[: args.top_k]:
        print(f"{f'({lo}, {hi})':>18} | {np.mean(ks_list):>8.4f}")

    best_range, best_ks = ranked[0][0], np.mean(ranked[0][1])
    print(f"\nBest fit: block_size_range = {tuple(best_range)}  (mean KS = {best_ks:.4f} over {args.n_data_seeds} data seeds)")


if __name__ == "__main__":
    main()
