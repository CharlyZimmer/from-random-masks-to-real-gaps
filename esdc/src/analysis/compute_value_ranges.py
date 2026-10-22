"""
Compute ESDC variable min/max ranges and write to computed_ranges.json.

Processes one zarr chunk at a time with inter-chunk sleeps to avoid HTTP 429
rate-limit errors. Zarr v3's codec pipeline uses asyncio.gather() internally,
ignoring the async.concurrency setting when fetching multiple chunks. By
requesting exactly one chunk-aligned slice per call, asyncio.gather() receives
a single item and issues exactly one HTTP GET.

Saves results after each variable so the script is fully resumable on failure.
Run this once before create_pretraining_dataset.sh; ensure_computed_ranges()
will then find all variables cached and skip the network computation.
"""
import argparse
import itertools
import json
import os
import time
from datetime import datetime

import fsspec
from fsspec.config import conf as _fsspec_conf
import numpy as np
import xarray as xr
import zarr

_fsspec_conf["nofiles_gather_batch_size"] = 1
_fsspec_conf["gather_batch_size"] = 1
if hasattr(zarr, "config"):
    zarr.config.set({"async.concurrency": 1})

ESDC_URL = (
    "https://data.rsc4earth.de/download/EarthSystemDataCube/v2.1.1/"
    "esdc-8d-0.083deg-184x270x270-2.1.1.zarr"
)

DEFAULT_VARS = [
    "land_surface_temperature",
    "air_temperature_2m",
    "leaf_area_index",
    "surface_moisture",
]


def _span_days(time_start: str, time_end: str) -> int:
    fmt = "%Y-%m-%d"
    return (datetime.strptime(time_end, fmt) - datetime.strptime(time_start, fmt)).days


def _load_ranges(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_ranges(path: str, ranges: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ranges, f, indent=2)


def _chunk_slices(da: xr.DataArray) -> list:
    """Return one isel-kwargs dict per zarr chunk, derived from da.chunks.

    xr.open_zarr (without explicit chunks=) uses the zarr array's native chunk
    shape as dask chunk sizes, so these slices align exactly with zarr storage
    chunks — each da.isel(one_slice).values triggers exactly one HTTP GET.
    """
    dim_slices = []
    for dim, chunk_sizes in zip(da.dims, da.chunks):
        boundaries = [0]
        for sz in chunk_sizes:
            boundaries.append(boundaries[-1] + sz)
        dim_slices.append(
            [(dim, slice(boundaries[i], boundaries[i + 1])) for i in range(len(boundaries) - 1)]
        )
    return [
        {dim: sl for dim, sl in combo}
        for combo in itertools.product(*dim_slices)
    ]


def _compute_var_range(
    da: xr.DataArray,
    var: str,
    inter_chunk_sleep: float,
) -> tuple:
    slices = _chunk_slices(da)
    total = len(slices)
    global_min = np.inf
    global_max = -np.inf
    t0 = time.monotonic()

    for idx, indexers in enumerate(slices):
        chunk = da.isel(indexers).values
        valid = chunk[np.isfinite(chunk)]
        if valid.size > 0:
            local_min = float(valid.min())
            local_max = float(valid.max())
            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max

        elapsed = time.monotonic() - t0
        done = idx + 1
        eta = (elapsed / done) * (total - done) if done < total else 0.0
        print(
            f"  [{var}] chunk {done}/{total}  "
            f"running [{global_min:.4g}, {global_max:.4g}]  "
            f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
            flush=True,
        )

        if idx < total - 1:
            time.sleep(inter_chunk_sleep)

    if not np.isfinite(global_min):
        raise ValueError(f"{var}: no finite values found across all chunks.")
    return global_min, global_max


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute ESDC variable min/max ranges one zarr chunk at a time "
            "and write results to computed_ranges.json."
        )
    )
    p.add_argument(
        "--vars", nargs="+", default=DEFAULT_VARS, metavar="VAR",
        help="Variable names to compute (default: all 4 standard ESDC vars).",
    )
    p.add_argument("--time_start", default="2002-05-29", help="Start of time range (ISO date).")
    p.add_argument("--time_end", default="2010-12-31", help="End of time range (ISO date).")
    p.add_argument(
        "--output",
        default=os.path.join(
            os.environ.get("ESDC_COMPUTED_RANGES_DIR", "esdc/data"),
            "computed_ranges.json",
        ),
        help="Path to computed_ranges.json. Default: $ESDC_COMPUTED_RANGES_DIR/computed_ranges.json.",
    )
    p.add_argument(
        "--inter_chunk_sleep", type=float, default=2.0,
        help="Seconds to sleep between zarr chunk fetches (default: 2.0).",
    )
    p.add_argument(
        "--inter_var_sleep", type=float, default=60.0,
        help="Seconds to sleep between variables (default: 60.0).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Recompute even if the variable is already cached with sufficient span.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    requested_days = _span_days(args.time_start, args.time_end)

    ranges = _load_ranges(args.output)
    print(f"Loaded {len(ranges)} existing entries from {args.output}")

    if args.force:
        need_compute = list(args.vars)
    else:
        need_compute = [
            v for v in args.vars
            if v not in ranges
            or _span_days(ranges[v]["time_start"], ranges[v]["time_end"]) < requested_days
        ]

    skipped = [v for v in args.vars if v not in need_compute]
    if skipped:
        print(f"Skipping (already covered): {skipped}")
    if not need_compute:
        print("All variables already have sufficient ranges. Use --force to recompute.")
        return

    print(f"Will compute: {need_compute}")
    print(f"Time span: {args.time_start} → {args.time_end}")
    print(f"Output: {args.output}")
    print(f"inter_chunk_sleep={args.inter_chunk_sleep}s  inter_var_sleep={args.inter_var_sleep}s")

    print("Opening ESDC zarr store ...")
    fs = fsspec.filesystem("https")
    ds = xr.open_zarr(fs.get_mapper(ESDC_URL), consolidated=True)
    ds = ds[need_compute].sel(time=slice(args.time_start, args.time_end))

    for var_idx, var in enumerate(need_compute):
        if var_idx > 0:
            print(f"Sleeping {args.inter_var_sleep:.0f}s between variables ...")
            time.sleep(args.inter_var_sleep)

        da = ds[var]
        n_chunks = int(np.prod([len(c) for c in da.chunks]))
        print(f"\n--- {var} ({var_idx + 1}/{len(need_compute)}) ---")
        print(f"  shape={da.shape}  n_zarr_chunks={n_chunks}")

        min_val, max_val = _compute_var_range(da, var, args.inter_chunk_sleep)

        print(f"  RESULT: min={min_val:.6g}  max={max_val:.6g}")
        ranges[var] = {
            "time_start": args.time_start,
            "time_end": args.time_end,
            "computed_range": [min_val, max_val],
        }
        _save_ranges(args.output, ranges)
        print(f"  Saved to {args.output}")

    print(f"\nDone. {len(need_compute)} variable(s) written to {args.output}")


if __name__ == "__main__":
    main()
