""" Sample and download 5-step Sentinel-2 spatiotemporal patches with Google Earth Engine

This script samples time series of consecutive Sentinel-2 images around given locations.
Each location yields a sequence of 5 consecutive, cloud-filtered time steps, and patches
are saved in a folder structure similar to SSL4EO-S12.

All shared GEE utilities live in gee_utils.py.
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
    OverlapError,
    fix_random_seeds,
    _build_time_series_patches,
    get_collection,
    get_random_patches_rtree,
    get_random_patches_grid,
    save_patch,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path", type=str, default="./data/", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S2_SR_HARMONIZED", help="GEE collection name"
    )
    parser.add_argument(
        "--meta_cloud_name",
        type=str,
        default="CLOUDY_PIXEL_PERCENTAGE",
        help="meta data cloud percentage name (kept for compatibility; not used in new pipeline)",
    )
    parser.add_argument(
        "--cloud_pct", type=int, default=20, help="cloud percentage threshold (legacy; not used)"
    )
    # time series properties
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["2020-06-01", "2020-07-02"],
        help="global date range [start end] for Sentinel-2 time series",
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=["B2", "B3", "B4"],
        help="bands to download for each time step",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", help="data type for saved patches"
    )
    # sampler / location options
    parser.add_argument(
        "--num_cities", type=int, default=10000, help="number of cities to sample (GaussianSampler)"
    )
    parser.add_argument(
        "--std", type=int, default=50, help="std of gaussian distribution in km"
    )
    # download settings
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--log_freq", type=int, default=10, help="print frequency")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run (checked_locations.csv)"
    )
    # location source: match pre-sampled coordinates OR free sampling with rtree/grid
    parser.add_argument(
        "--match_file",
        type=str,
        default=None,
        help="CSV file with pre-sampled coordinates and indices (e.g. SSL4EO-S12 centers)",
    )
    parser.add_argument(
        "--overlap_check",
        type=str,
        default="rtree",
        choices=["grid", "rtree", None],
        help="overlap check method when sampling new locations",
    )
    parser.add_argument(
        "--indices_range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to process from match_file or for new sampling",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    # cloud / shadow / series parameters
    parser.add_argument("--patch_size", type=int, default=128, help="patch size in pixels (10 m)")
    parser.add_argument(
        "--max_gap_ratio",
        type=float,
        default=0.2,
        help="maximum allowed cloud/shadow fraction in the middle frame",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=5,
        help="number of consecutive time steps per sample",
    )
    parser.add_argument(
        "--cld_prb_thresh",
        type=int,
        default=50,
        help="s2cloudless probability threshold for clouds (percent)",
    )
    parser.add_argument(
        "--nir_drk_thresh",
        type=float,
        default=0.15,
        help="near-infrared reflectance threshold for dark pixels (potential shadows)",
    )
    parser.add_argument(
        "--cld_prj_dist",
        type=float,
        default=1.0,
        help="maximum distance (km) to search for cloud shadows from cloud edges",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=50,
        help="distance (m) to dilate the edge of cloud-identified objects",
    )
    parser.add_argument(
        "--ee_project",
        type=str,
        help="Earth Engine project ID",
    )
    parser.add_argument(
        "--ee_service_account",
        type=str,
        default=None,
        help="Service account email for Earth Engine (optional)",
    )
    parser.add_argument(
        "--ee_key_file",
        type=str,
        default=None,
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
    RADIUS = PATCH_SIZE * 10 / 2  # in meters
    COL_NAME = args.collection

    EE_PROJECT = args.ee_project
    EE_SERVICE_ACCOUNT = args.ee_service_account
    EE_KEY_FILE = args.ee_key_file

    # initialize ee
    if EE_SERVICE_ACCOUNT and EE_KEY_FILE:
        credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_KEY_FILE)
        ee.Initialize(credentials=credentials, project=EE_PROJECT)
    else:
        ee.Initialize(project=EE_PROJECT)

    # initialize sampler (only used when match_file is not provided)
    sampler = GaussianSampler(num_cities=args.num_cities, std=args.std)

    # parse date range for time series (multi-year span)
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
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)
                ext_flags[key] = int(row[3])
    else:
        os.makedirs(args.save_path, exist_ok=True)
        ext_path = os.path.join(args.save_path, "checked_locations.csv")
        if os.path.exists(ext_path):
            os.remove(ext_path)
        ssl_dir = os.path.join(args.save_path, "SSL-consecutive")
        if os.path.isdir(ssl_dir):
            shutil.rmtree(ssl_dir)
        print(f"Fresh run: cleared existing data at {args.save_path}")

    # location source: match from pre-sampled coords (e.g. SSL4EO-S12)
    if args.match_file:
        with open(args.match_file) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                match_coords[key] = (val1, val2)
        grid_dict: Dict[Any, Any] = {}
        rtree_coords = index.Index()
    elif args.overlap_check is not None:
        grid_dict: Dict[Any, Any] = {}
        rtree_coords = index.Index()
        if args.resume:
            print("Load existing locations.")
            for i, key in enumerate(tqdm(ext_coords.keys())):
                c = ext_coords[key]
                rtree_coords.insert(i, (c[0], c[1], c[0], c[1]))
                gridIndex = (math.floor(c[0] + 180), math.floor(c[1] + 90))
                if gridIndex not in grid_dict.keys():
                    grid_dict[gridIndex] = {c}
                else:
                    grid_dict[gridIndex].add(c)
    else:
        raise NotImplementedError

    start_time = time.time()
    counter = Counter()

    def worker(idx: int) -> None:
        # skip already processed indices
        if str(idx) in ext_coords.keys() and ext_flags.get(str(idx), 0) != 0:
            return

        # choose center coordinate
        if args.match_file:
            center_coord = match_coords[str(idx)]
        else:
            if args.overlap_check == "rtree":
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
            elif args.overlap_check == "grid":
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
            else:
                center_coord = sampler.sample_point()

        # draw a random 1-month window within the global date span, seeded
        # deterministically per idx (instead of drawing from the shared global
        # np.random state) so that:
        #  - the assigned window is invariant to worker thread interleaving
        #    (workers run as threads sharing one process, see Pool below), and
        #  - re-running this script (e.g. a non-resumed "fresh run", which
        #    wipes and rebuilds SSL-consecutive/) reassigns the exact same
        #    window to a given idx instead of a different one, which would
        #    silently invalidate any gap-filling training gap_masks/{idx}
        #    already built against that idx's window.
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

        if patches is not None:
            if args.save_path is not None:
                location_path = os.path.join(args.save_path, "SSL-consecutive", f"{idx:06d}")
                os.makedirs(location_path, exist_ok=True)
                for patch in patches:
                    save_patch(
                        raster=patch["raster"],
                        coords=patch["coords"],
                        metadata=patch["metadata"],
                        path=location_path,
                    )
                with open(os.path.join(location_path, "location.json"), "w") as _f:
                    json.dump(
                        {
                            "idx": idx,
                            "lon": center_coord[0],
                            "lat": center_coord[1],
                            "query_window": [win_start_str, win_end_str],
                        },
                        _f,
                    )

            count = counter.update(1)
            if count % args.log_freq == 0:
                print(f"Downloaded {count} time-series samples in {time.time() - start_time:.3f}s.")
        else:
            print("no suitable 5-step time series for location %d." % (idx))

        # log to checked locations
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            if patches is not None:
                success = 1 if not args.match_file else 2
            else:
                success = 0
            data = [idx, center_coord[0], center_coord[1], success]
            writer.writerow(data)

    # set indices
    if args.match_file is not None:
        indices: List[int] = []
        for key in match_coords.keys():
            indices.append(int(key))
        indices = indices[args.indices_range[0] : args.indices_range[1]]
    elif args.indices_range is not None:
        indices = list(range(args.indices_range[0], args.indices_range[1]))
    else:
        print("Please set up indices.")
        raise NotImplementedError

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
