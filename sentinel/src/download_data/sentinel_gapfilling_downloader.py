"""Download a Sentinel-2 gap-filling dataset.

For each sampled location this script:
  1. Downloads a 5-step consecutive time series whose middle frame contains
     at most 20% cloud/shadow gap pixels (max_gap_ratio=0.2, matching ESDC's
     gap-filling eval dataset). Locations are either matched from a
     pre-sampled coordinates file (e.g. the SSL4EO-S12 pre-training centers,
     via --match_file — used for the gap-filling *training* set, which reuses
     the pre-training locations) or freely Gaussian-sampled (no --match_file
     — used for the gap-filling *evaluation* set, which stays disjoint from
     the pre-training locations).
  2. Finds a *donor* image from the same 1-month window (not one of the 5
     selected timesteps) and extracts its binary cloud/shadow mask.
  3. Applies that mask as a synthetic gap to the clean middle frame.
  4. Saves the modified 5-step series under  <save_path>/SSL-consecutive/
     (standard format, compatible with create_LMDB.py) and the original clean
     middle frame + binary gap mask under  <save_path>/ground_truth/.

Output layout
-------------
<save_path>/
├── SSL-consecutive/               # model input
│   └── {idx:06d}/
│       ├── {t0_id}/S2C.tif  metadata.json
│       ├── {t1_id}/...
│       ├── {t2_id}/               # middle frame — NaN where gap_mask==True
│       │   ├── S2C.tif
│       │   └── metadata.json
│       ├── {t3_id}/...
│       └── {t4_id}/...
└── ground_truth/                  # evaluation reference
    └── {idx:06d}/
        └── {t2_id}/
            ├── S2C.tif            (all 10 bands, original, no NaN)
            ├── gap_mask.tif       (uint8, 1 = synthetically gapped pixel)
            └── metadata.json
"""

import argparse
import csv
import json
import math
import os
import shutil
import time
from datetime import date, timedelta
from multiprocessing.dummy import Pool
from typing import Any, Dict, List, Optional, Tuple

import ee
import numpy as np
from rtree import index
from tqdm import tqdm

from sentinel.src.download_data.gee_utils import (
    GaussianSampler,
    Counter,
    fix_random_seeds,
    _build_time_series_patches,
    _get_donor_gap_map,
    get_collection,
    get_random_patches_rtree,
    get_random_patches_grid,
    save_patch,
    save_geotiff,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download a Sentinel-2 gap-filling dataset (training or evaluation)"
    )
    parser.add_argument(
        "--save_path", type=str, default="./data/gap_filling", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S2_SR_HARMONIZED",
        help="GEE collection name",
    )
    parser.add_argument(
        "--meta_cloud_name", type=str, default="CLOUDY_PIXEL_PERCENTAGE",
        help="metadata cloud percentage name (used by legacy overlap-check step only)",
    )
    parser.add_argument(
        "--cloud_pct", type=int, default=20,
        help="cloud percentage threshold (legacy overlap-check step only)",
    )
    # time series properties
    parser.add_argument(
        "--dates", type=str, nargs="+", default=["2019-01-01", "2025-12-31"],
        help="global date range [start end] for Sentinel-2 time series",
    )
    parser.add_argument(
        "--bands", type=str, nargs="+", default=["B2", "B3", "B4"],
        help="bands to download for each time step",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", help="data type for saved patches"
    )
    # sampler / location options
    parser.add_argument(
        "--num_cities", type=int, default=10000,
        help="number of top cities for GaussianSampler",
    )
    parser.add_argument(
        "--std", type=int, default=50, help="std of Gaussian distribution in km"
    )
    # location source: match pre-sampled coordinates OR free Gaussian sampling
    parser.add_argument(
        "--match_file",
        type=str,
        default=None,
        help="CSV file with pre-sampled coordinates and indices (e.g. SSL4EO-S12 centers)",
    )
    parser.add_argument(
        "--overlap_check", type=str, default="rtree", choices=["grid", "rtree"],
        help="overlap check method for location sampling (used only when --match_file is not given)",
    )
    # download settings
    parser.add_argument("--num_workers", type=int, default=8, help="number of parallel workers")
    parser.add_argument("--log_freq", type=int, default=10, help="log every N successful samples")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="resume from a previous run (checked_locations.csv)",
    )
    parser.add_argument(
        "--indices_range", type=int, nargs=2, default=[0, 250000],
        help="indices to process from match_file or for new sampling",
    )
    parser.add_argument("--debug", action="store_true", help="print verbose GEE errors")
    # patch / cloud / shadow parameters
    parser.add_argument(
        "--patch_size", type=int, default=128, help="patch size in pixels (10 m resolution)"
    )
    parser.add_argument(
        "--max_gap_ratio", type=float, default=0.2,
        help="maximum allowed cloud/shadow fraction in the middle frame before synthetic-gap injection, matching ESDC's gap-filling eval dataset",
    )
    parser.add_argument(
        "--n_timesteps", type=int, default=5,
        help="number of consecutive time steps per sample",
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
    # synthetic gap parameters
    parser.add_argument(
        "--gap_donor_min_ratio", type=float, default=0.05,
        help="minimum cloud fraction of donor image for synthetic gap (inclusive)",
    )
    parser.add_argument(
        "--gap_donor_max_ratio", type=float, default=0.85,
        help="maximum cloud fraction of donor image for synthetic gap (inclusive)",
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

    WINDOW_SEED = 42
    fix_random_seeds(seed=WINDOW_SEED)

    PATCH_SIZE = args.patch_size
    MAX_GAP_RATIO = args.max_gap_ratio
    N_TIMESTEPS = args.n_timesteps
    CLD_PRB_THRESH = args.cld_prb_thresh
    NIR_DRK_THRESH = args.nir_drk_thresh
    CLD_PRJ_DIST = args.cld_prj_dist
    BUFFER = args.buffer
    RADIUS = PATCH_SIZE * 10 / 2  # metres
    COL_NAME = args.collection

    # initialize Earth Engine
    if args.ee_service_account and args.ee_key_file:
        credentials = ee.ServiceAccountCredentials(args.ee_service_account, args.ee_key_file)
        ee.Initialize(credentials=credentials, project=args.ee_project)
    else:
        ee.Initialize(project=args.ee_project)

    # initialize sampler (only used when match_file is not provided)
    sampler = GaussianSampler(num_cities=args.num_cities, std=args.std)

    # parse global date range
    if len(args.dates) >= 2:
        global_start_date = min(args.dates)
        global_end_date = max(args.dates)
    else:
        global_start_date = args.dates[0]
        global_end_date = args.dates[0]

    bands = args.bands
    dtype = args.dtype

    # resume file bookkeeping
    ext_coords: Dict[str, Tuple[float, float]] = {}
    ext_flags: Dict[str, int] = {}
    match_coords: Dict[str, Tuple[float, float]] = {}
    if args.resume:
        ext_path = args.resume
        with open(ext_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                ext_coords[key] = (float(row[1]), float(row[2]))
                ext_flags[key] = int(row[3])
    else:
        os.makedirs(args.save_path, exist_ok=True)
        ext_path = os.path.join(args.save_path, "checked_locations.csv")
        if os.path.exists(ext_path):
            os.remove(ext_path)
        for subdir in ("SSL-consecutive", "ground_truth"):
            d = os.path.join(args.save_path, subdir)
            if os.path.isdir(d):
                shutil.rmtree(d)
        print(f"Fresh run: cleared existing data at {args.save_path}")

    # location source: match from pre-sampled coords (e.g. SSL4EO-S12), or
    # build the overlap-check data structure for free Gaussian sampling
    grid_dict: Dict[Any, Any] = {}
    rtree_coords = index.Index()
    if args.match_file:
        with open(args.match_file) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                match_coords[key] = (float(row[1]), float(row[2]))
    elif args.resume:
        print("Loading existing locations into overlap checker...")
        for i, key in enumerate(tqdm(ext_coords.keys())):
            c = ext_coords[key]
            rtree_coords.insert(i, (c[0], c[1], c[0], c[1]))
            gridIndex = (math.floor(c[0] + 180), math.floor(c[1] + 90))
            if gridIndex not in grid_dict:
                grid_dict[gridIndex] = {c}
            else:
                grid_dict[gridIndex].add(c)

    start_time = time.time()
    counter = Counter()

    def worker(idx: int) -> None:
        # skip already processed indices
        if str(idx) in ext_coords and ext_flags.get(str(idx), 0) != 0:
            return

        # choose center coordinate
        if args.match_file:
            center_coord = match_coords[str(idx)]
        elif args.overlap_check == "rtree":
            _, center_coord = get_random_patches_rtree(
                idx,
                get_collection(args.collection, args.meta_cloud_name, args.cloud_pct),
                bands,
                {},
                dtype,
                sampler,
                [],
                radius=RADIUS,
                debug=args.debug,
                rtree_obj=rtree_coords,
            )
        else:  # grid
            _, center_coord = get_random_patches_grid(
                idx,
                get_collection(args.collection, args.meta_cloud_name, args.cloud_pct),
                bands,
                {},
                dtype,
                sampler,
                [],
                radius=RADIUS,
                debug=args.debug,
                grid_dict=grid_dict,
            )

        # draw a random 1-month window within the global date span, seeded
        # deterministically per idx (instead of drawing from the shared global
        # np.random state) so that the assigned window is invariant to worker
        # thread interleaving (workers run as threads sharing one process, see
        # Pool below) and to re-running this script -- see the equivalent comment
        # in sentinel_spatiotemporal_downloader.py for the full rationale.
        start_dt = date.fromisoformat(global_start_date)
        end_dt = date.fromisoformat(global_end_date)
        span_days = (end_dt - start_dt).days
        if span_days <= 30:
            win_start_dt = start_dt
        else:
            idx_rng = np.random.default_rng([WINDOW_SEED, idx])
            offset = int(idx_rng.integers(0, span_days - 30))
            win_start_dt = start_dt + timedelta(days=offset)
        win_end_dt = win_start_dt + timedelta(days=30)
        win_start_str = win_start_dt.strftime("%Y-%m-%d")
        win_end_str = win_end_dt.strftime("%Y-%m-%d")

        # download 5-step time series with a fully clean middle frame
        patches, center_coord = _build_time_series_patches(
            list(center_coord),
            start_date=win_start_str,
            end_date=win_end_str,
            bands=bands,
            radius_m=RADIUS,
            n_timesteps=N_TIMESTEPS,
            max_gap_ratio=MAX_GAP_RATIO,
            cld_prb_thresh=CLD_PRB_THRESH,
            nir_drk_thresh=NIR_DRK_THRESH,
            cld_prj_dist=CLD_PRJ_DIST,
            buffer_m=BUFFER,
            col_name=COL_NAME,
            patch_size=PATCH_SIZE,
            dtype=dtype,
        )

        success = 0

        if patches is None:
            print(f"No suitable 5-step clean time series for location {idx}.")
        else:
            # IDs of the 5 selected window images (used to exclude them from donor pool)
            exclude_ids = [
                p["metadata"]["properties"]["system:index"] for p in patches
            ]

            # find a donor image from the same 1-month window for synthetic gap injection
            gap_mask, gap_fraction = _get_donor_gap_map(
                coords=list(center_coord),
                start_date=win_start_str,
                end_date=win_end_str,
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
                idx=idx,
            )

            if gap_mask is None:
                print(
                    f"No suitable donor gap map for location {idx} "
                    f"(window {win_start_str}–{win_end_str}); skipping."
                )
            else:
                # preserve the clean middle frame before modifying it
                middle_patch = patches[2]  # index 2 of 5
                gt_raster = {b: arr.copy() for b, arr in middle_patch["raster"].items()}

                # inject synthetic gap into the middle frame
                for arr in middle_patch["raster"].values():
                    arr[gap_mask] = np.nan

                # --- save gappy 5-step input (standard SSL-consecutive format) ---
                location_path = os.path.join(
                    args.save_path, "SSL-consecutive", f"{idx:06d}"
                )
                os.makedirs(location_path, exist_ok=True)
                for patch in patches:
                    save_patch(
                        raster=patch["raster"],
                        coords=patch["coords"],
                        metadata=patch["metadata"],
                        path=location_path,
                    )
                with open(os.path.join(location_path, "location.json"), "w") as _f:
                    json.dump({"idx": idx, "lon": center_coord[0], "lat": center_coord[1]}, _f)

                # --- save ground truth (clean middle frame + binary gap mask) ---
                gt_location_dir = os.path.join(
                    args.save_path, "ground_truth", f"{idx:06d}"
                )
                save_patch(
                    raster=gt_raster,
                    coords=middle_patch["coords"],
                    metadata=middle_patch["metadata"],
                    path=gt_location_dir,
                )

                # add gap_mask.tif to the same subfolder that save_patch just created
                middle_id = middle_patch["metadata"]["properties"]["system:index"]
                gt_patch_dir = os.path.join(gt_location_dir, middle_id)
                mask_img = gap_mask.astype(np.uint8)[..., np.newaxis]  # (H, W, 1)
                save_geotiff(
                    mask_img,
                    middle_patch["coords"],
                    os.path.join(gt_patch_dir, "gap_mask.tif"),
                )

                count = counter.update(1)
                if count % args.log_freq == 0:
                    print(
                        f"Downloaded {count} gap-filling samples "
                        f"(donor gap fraction: {gap_fraction:.2f}) "
                        f"in {time.time() - start_time:.3f}s."
                    )
                success = 1

        # log to checked_locations
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([idx, center_coord[0], center_coord[1], success])

    # build index list
    if args.match_file:
        indices: List[int] = [int(key) for key in match_coords.keys()]
        indices = indices[args.indices_range[0] : args.indices_range[1]]
    else:
        indices = list(range(args.indices_range[0], args.indices_range[1]))

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
