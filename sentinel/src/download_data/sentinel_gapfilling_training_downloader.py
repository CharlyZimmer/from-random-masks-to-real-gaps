"""Download synthetic gap masks for the Sentinel-2 gap-filling *training* set.

Unlike sentinel_gapfilling_downloader.py (which independently re-downloads a
full 5-step time series per location -- used for the spatially-disjoint,
Gaussian-sampled *evaluation* set), this script reuses the pretraining
dataset's already-downloaded imagery for the gap-filling *training* set. The
two datasets share the same SSL4EO-S12 locations (same idx numbering), so
there is no need to store the 5-step series or a clean ground-truth frame a
second time -- only the small synthetic gap mask is new information.

For each idx already present under --pretrain_dir/SSL-consecutive:
  1. Read the existing 5-step series' metadata (system:index, system:time_start
     per timestep) -- no GEE query needed for the main series.
  2. Check the reused middle frame's pre-existing gap fraction against
     --max_gap_ratio (default 0.2, matching ESDC's gap-filling training
     dataset); locations exceeding it are skipped.
  3. Find a *donor* image in roughly the same time window (excluded from the
     5 stored timesteps by system:index) and extract its binary cloud/shadow
     mask, via the same _get_donor_gap_map() used by the eval-set downloader.
  4. Save only the binary gap mask (+ small metadata) under
     <save_path>/gap_masks/<idx>/gap_mask.tif.

The gappy input series and the ground-truth middle frame are both
reconstructed at training time from the pretraining LMDB + this gap mask
(see model_gap_aware/lmdb_training.py::PretrainGapMaskJoinDataset) -- they are
never materialized on disk for the training set.

Output layout
-------------
<save_path>/
└── gap_masks/
    └── {idx:06d}/
        ├── gap_mask.tif   (uint8, 1 = synthetically gapped pixel)
        └── metadata.json  (idx, gap_fraction, donor search window, middle system:index)
"""

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter as _ReasonCounter
from datetime import datetime, timedelta
from multiprocessing.dummy import Lock, Pool
from typing import Any, Dict, List, Optional, Tuple

import ee
import numpy as np
import rasterio

from sentinel.src.download_data.gee_utils import (
    Counter,
    fix_random_seeds,
    _get_donor_gap_map,
    save_geotiff,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download synthetic gap masks for the Sentinel-2 gap-filling training set"
    )
    parser.add_argument(
        "--save_path", type=str, default="./data/gap_filling_training", help="dir to save data"
    )
    parser.add_argument(
        "--pretrain_dir", type=str, required=True,
        help="Root of the pretraining dataset (contains SSL-consecutive/{idx:06d}/...)",
    )
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S2_SR_HARMONIZED",
        help="GEE collection name",
    )
    parser.add_argument(
        "--bands", type=str, nargs="+",
        default=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        help="bands to download for the donor cloud mask",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", help="data type used when downloading the donor patch"
    )
    parser.add_argument(
        "--match_file", type=str, default=None,
        help="Fallback CSV of idx,lon,lat, used only if a location's location.json is missing",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="number of parallel workers")
    parser.add_argument("--log_freq", type=int, default=10, help="log every N successful samples")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="resume from a previous run (checked_locations.csv)",
    )
    parser.add_argument(
        "--indices_range", type=int, nargs=2, default=[0, 251078],
        help="idx range to process (must already exist under --pretrain_dir/SSL-consecutive)",
    )
    parser.add_argument("--debug", action="store_true", help="print verbose GEE errors")
    parser.add_argument(
        "--patch_size", type=int, default=128, help="patch size in pixels (10 m resolution)"
    )
    parser.add_argument(
        "--n_timesteps", type=int, default=5,
        help="expected number of consecutive time steps per pretraining sample (sanity check)",
    )
    parser.add_argument(
        "--max_gap_ratio", type=float, default=0.2,
        help="maximum allowed pre-existing gap fraction in the reused pretraining middle "
             "frame, matching ESDC's gap-filling training dataset",
    )
    parser.add_argument(
        "--cld_prb_thresh", type=int, default=50,
        help="s2cloudless probability threshold for clouds (percent)",
    )
    parser.add_argument(
        "--nir_drk_thresh", type=float, default=0.15,
        help="near-infrared reflectance threshold for dark pixels (potential shadows)",
    )
    parser.add_argument(
        "--cld_prj_dist", type=float, default=1.0,
        help="maximum distance (km) to project cloud shadows",
    )
    parser.add_argument(
        "--buffer", type=int, default=50,
        help="distance (m) to dilate cloud/shadow mask edges",
    )
    parser.add_argument(
        "--gap_donor_min_ratio", type=float, default=0.05,
        help="minimum cloud fraction of donor image for synthetic gap (inclusive)",
    )
    parser.add_argument(
        "--gap_donor_max_ratio", type=float, default=0.85,
        help="maximum cloud fraction of donor image for synthetic gap (inclusive)",
    )
    parser.add_argument(
        "--window_pad_days", type=int, default=15,
        help="days to pad the donor search window on each side when a sample's "
             "location.json lacks a persisted query_window (older pretraining downloads)",
    )
    parser.add_argument(
        "--donor_window_expand_days", type=int, default=60,
        help="expand the donor search window symmetrically by N days on each side "
             "(applied to both the persisted query_window and the --window_pad_days "
             "fallback window). Pretraining already consumed ~5 of the ~6 acquisitions "
             "in the base window, so the leftover donor pool is near-empty without this.",
    )
    parser.add_argument(
        "--donor_probe_k", type=int, default=20,
        help="probe the donor gap fraction (a reduceRegion per candidate) for only "
             "the K candidates temporally closest to the reused 5-step window, so the "
             "per-sample GEE cost stays bounded regardless of --donor_window_expand_days",
    )
    # Earth Engine authentication
    parser.add_argument("--ee_project", type=str, help="Earth Engine project ID")
    parser.add_argument(
        "--ee_service_account", type=str, default=None,
        help="Service account email for Earth Engine (optional)",
    )
    parser.add_argument(
        "--ee_key_file", type=str, default=None,
        help="Path to service account JSON key file (optional)",
    )

    args = parser.parse_args()

    fix_random_seeds(seed=42)

    PATCH_SIZE = args.patch_size
    N_TIMESTEPS = args.n_timesteps
    CLD_PRB_THRESH = args.cld_prb_thresh
    NIR_DRK_THRESH = args.nir_drk_thresh
    CLD_PRJ_DIST = args.cld_prj_dist
    BUFFER = args.buffer
    RADIUS = PATCH_SIZE * 10 / 2  # metres
    COL_NAME = args.collection
    bands = args.bands
    dtype = args.dtype

    # initialize Earth Engine
    if args.ee_service_account and args.ee_key_file:
        credentials = ee.ServiceAccountCredentials(args.ee_service_account, args.ee_key_file)
        ee.Initialize(credentials=credentials, project=args.ee_project)
    else:
        ee.Initialize(project=args.ee_project)

    # optional fallback coordinates (only used if a location.json is missing)
    match_coords: Dict[str, Tuple[float, float]] = {}
    if args.match_file:
        with open(args.match_file) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                match_coords[row[0]] = (float(row[1]), float(row[2]))

    # resume file bookkeeping
    ext_flags: Dict[str, int] = {}
    if args.resume:
        # Resume run: keep every previously-written gap_masks/{idx}/ and only
        # re-attempt indices that were skipped last time. Never clear anything.
        ext_path = args.resume
        os.makedirs(args.save_path, exist_ok=True)
        if os.path.exists(ext_path):
            with open(ext_path) as csv_file:
                for row in csv.reader(csv_file):
                    if row:
                        ext_flags[row[0]] = int(row[3])
            print(
                f"Resume run: {len(ext_flags)} locations already checked; "
                f"existing output under {args.save_path} left intact."
            )
        else:
            print(
                f"Resume run: {ext_path} not found; starting fresh but NOT "
                f"clearing {args.save_path}."
            )
    else:
        os.makedirs(args.save_path, exist_ok=True)
        ext_path = os.path.join(args.save_path, "checked_locations.csv")
        if os.path.exists(ext_path):
            os.remove(ext_path)
        # clear any stale old-layout (or previous-run) output so it can't be
        # silently reused alongside newly-written samples
        for subdir in ("SSL-consecutive", "ground_truth", "gap_masks"):
            d = os.path.join(args.save_path, subdir)
            if os.path.isdir(d):
                shutil.rmtree(d)
        print(f"Fresh run: cleared existing data at {args.save_path}")

    pretrain_ssl_dir = os.path.join(args.pretrain_dir, "SSL-consecutive")

    start_time = time.time()
    counter = Counter()
    reasons = _ReasonCounter()
    reasons_lock = Lock()

    def _tally(reason: str) -> None:
        with reasons_lock:
            reasons[reason] += 1

    def worker(idx: int) -> None:
        key = str(idx)
        if key in ext_flags and ext_flags[key] != 0:
            _tally("already_done")
            return

        gm_dir = os.path.join(args.save_path, "gap_masks", f"{idx:06d}")
        if os.path.exists(os.path.join(gm_dir, "gap_mask.tif")):
            # idempotent: a resume run must never redo (or wipe) a finished idx
            _tally("already_done")
            return

        def _skip(reason: str, lon: float = 0.0, lat: float = 0.0) -> None:
            _tally(reason)
            with open(ext_path, "a") as f:
                csv.writer(f).writerow([idx, lon, lat, 0, reason])

        loc_dir = os.path.join(pretrain_ssl_dir, f"{idx:06d}")
        if not os.path.isdir(loc_dir):
            _skip("pretrain_dir_missing")
            return

        loc_json_path = os.path.join(loc_dir, "location.json")
        query_window: Optional[List[str]] = None
        if os.path.exists(loc_json_path):
            with open(loc_json_path) as f:
                loc_info = json.load(f)
            lon, lat = float(loc_info["lon"]), float(loc_info["lat"])
            query_window = loc_info.get("query_window")
        elif key in match_coords:
            lon, lat = match_coords[key]
        else:
            _skip("no_location_json")
            return

        time_steps = sorted(
            d for d in os.listdir(loc_dir) if os.path.isdir(os.path.join(loc_dir, d))
        )
        if len(time_steps) != N_TIMESTEPS:
            print(f"Location {idx}: expected {N_TIMESTEPS} time steps, found {len(time_steps)}; skipping.")
            _skip("wrong_n_timesteps", lon, lat)
            return

        exclude_ids: List[str] = []
        time_starts: List[int] = []
        for t in time_steps:
            with open(os.path.join(loc_dir, t, "metadata.json")) as f:
                meta = json.load(f)
            props = meta["properties"]
            exclude_ids.append(props["system:index"])
            time_starts.append(int(props["system:time_start"]))

        mid_idx = N_TIMESTEPS // 2
        mid_dir = os.path.join(loc_dir, time_steps[mid_idx])
        with open(os.path.join(mid_dir, "metadata.json")) as f:
            mid_meta = json.load(f)
        middle_system_index = mid_meta["properties"]["system:index"]
        patch_coords = mid_meta["coords"]

        with rasterio.open(os.path.join(mid_dir, "S2C.tif")) as src:
            mid_gap_fraction = float(np.isnan(src.read(1)).mean())
        if mid_gap_fraction > args.max_gap_ratio:
            print(
                f"Location {idx}: middle-frame gap fraction {mid_gap_fraction:.2f} "
                f"exceeds --max_gap_ratio {args.max_gap_ratio}; skipping."
            )
            _skip("middle_gap_exceeds", lon, lat)
            return

        if query_window:
            win_start_str, win_end_str = query_window
        else:
            min_date = datetime.utcfromtimestamp(min(time_starts) / 1000.0).date()
            max_date = datetime.utcfromtimestamp(max(time_starts) / 1000.0).date()
            win_start_str = (min_date - timedelta(days=args.window_pad_days)).strftime("%Y-%m-%d")
            win_end_str = (max_date + timedelta(days=args.window_pad_days)).strftime("%Y-%m-%d")

        # Widen the donor search window. Pretraining's _build_time_series_patches
        # already took ~5 of the ~6 deduped acquisitions in [win_start, win_end],
        # so the leftover donor pool here is near-empty; expanding it symmetrically
        # restores a usable pool while keeping the donor within
        # --donor_window_expand_days of the reused window (same season).
        _exp = timedelta(days=args.donor_window_expand_days)
        donor_win_start = (datetime.strptime(win_start_str, "%Y-%m-%d") - _exp).strftime("%Y-%m-%d")
        donor_win_end = (datetime.strptime(win_end_str, "%Y-%m-%d") + _exp).strftime("%Y-%m-%d")

        gap_mask, gap_fraction = _get_donor_gap_map(
            coords=[lon, lat],
            start_date=donor_win_start,
            end_date=donor_win_end,
            exclude_system_indexes=exclude_ids,
            bands=bands,
            radius_m=RADIUS,
            patch_size=PATCH_SIZE,
            cld_prb_thresh=CLD_PRB_THRESH,
            nir_drk_thresh=NIR_DRK_THRESH,
            cld_prj_dist=CLD_PRJ_DIST,
            buffer_m=BUFFER,
            col_name=COL_NAME,
            min_gap_ratio=args.gap_donor_min_ratio,
            max_gap_ratio=args.gap_donor_max_ratio,
            dtype=dtype,
            window_time_starts=time_starts,
            donor_probe_k=args.donor_probe_k,
            idx=idx,
        )

        if gap_mask is None:
            print(
                f"No suitable donor gap map for location {idx} "
                f"(donor window {donor_win_start}–{donor_win_end}); skipping."
            )
            _skip("no_donor", lon, lat)
            return

        os.makedirs(gm_dir, exist_ok=True)
        mask_img = gap_mask.astype(np.uint8)[..., np.newaxis]  # (H, W, 1)
        save_geotiff(mask_img, patch_coords, os.path.join(gm_dir, "gap_mask.tif"))
        with open(os.path.join(gm_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "idx": idx,
                    "gap_fraction": gap_fraction,
                    "mid_gap_fraction": mid_gap_fraction,
                    "window_start": win_start_str,
                    "window_end": win_end_str,
                    "donor_window_start": donor_win_start,
                    "donor_window_end": donor_win_end,
                    "middle_system_index": middle_system_index,
                },
                f,
            )

        _tally("success")
        count = counter.update(1)
        if count % args.log_freq == 0:
            print(
                f"Downloaded {count} gap masks (donor gap fraction: {gap_fraction:.2f}) "
                f"in {time.time() - start_time:.3f}s."
            )

        with open(ext_path, "a") as f:
            csv.writer(f).writerow([idx, lon, lat, 1, "success"])

    indices: List[int] = list(range(args.indices_range[0], args.indices_range[1]))

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)

    elapsed = time.time() - start_time
    total = len(indices)
    order = [
        "success", "already_done", "pretrain_dir_missing", "no_location_json",
        "wrong_n_timesteps", "middle_gap_exceeds", "no_donor",
    ]
    print("\n=== gap-mask download summary ===")
    for reason in order:
        print(f"  {reason:22s}: {reasons.get(reason, 0):>8d}")
    for reason in sorted(set(reasons) - set(order)):
        print(f"  {reason:22s}: {reasons[reason]:>8d}")
    print(f"  {'TOTAL processed':22s}: {total:>8d}   in {elapsed / 3600:.2f} h")
    have_imagery = total - reasons.get("pretrain_dir_missing", 0) - reasons.get("no_location_json", 0)
    if have_imagery > 0:
        got = reasons.get("success", 0) + reasons.get("already_done", 0)
        print(f"  success rate over idx with pretraining imagery: {got / have_imagery:.1%}")
