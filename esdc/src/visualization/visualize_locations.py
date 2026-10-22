"""Save a global density map of ESDC sample coverage to img/esdc_locations.png.

Each sample contributes +1 count to every grid cell within its 64×64 pixel footprint,
so the map shows how often specific world-map pixels are represented in the dataset.
"""
import argparse
import json
import os

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pyogrio
import cmocean

_DEFAULT_DATA_DIR = os.environ.get("ESDC_DATA_DIR", "esdc/data/lst_no_syn_gaps")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
OUT_PATH = os.path.join(_REPO_ROOT, "img", "esdc_locations.png")

_WORLD_SHP = os.path.join(
    os.path.dirname(pyogrio.__file__),
    "tests", "fixtures", "naturalearth_lowres", "naturalearth_lowres.shp",
)


def _load_coords(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    h5_path = os.path.join(data_dir, "dataset.h5")
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        lats = [f["lat_train"][:]]
        lons = [f["lon_train"][:]]
        if "lat_test" in keys:
            lats.append(f["lat_test"][:])
            lons.append(f["lon_test"][:])
    return np.concatenate(lats), np.concatenate(lons)


def _load_meta(data_dir: str) -> dict:
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path) as fh:
        return json.load(fh)


def _derive_pixel_res(lat: np.ndarray, stride_lat: int, lon: np.ndarray, stride_lon: int) -> tuple[float, float]:
    """Estimate pixel resolution from the spacing between unique patch centers."""
    lat_sorted = np.sort(np.unique(np.round(lat.astype(float), 4)))
    lon_sorted = np.sort(np.unique(np.round(lon.astype(float), 4)))
    lat_res = float(np.median(np.diff(lat_sorted))) / stride_lat
    lon_res = float(np.median(np.diff(lon_sorted))) / stride_lon
    return lat_res, lon_res


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="Directory containing dataset.h5 and meta.json.")
    p.add_argument("--out", default=OUT_PATH, help="Output PNG path.")
    p.add_argument("--title", default="ESDC Pre-Training Dataset", help="Dataset label used in the plot title.")
    p.add_argument("--vmax", type=int, default=None,
                   help="Maximum sample count for the color scale (default: use observed max).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lat, lon = _load_coords(args.data_dir)
    print(f"Loaded {len(lat):,} samples")

    meta = _load_meta(args.data_dir)
    block_h = meta["block_size"][1]  # 64
    block_w = meta["block_size"][2]  # 64
    patch_overlap = meta.get("patch_overlap", 0)
    strides = meta["strides"]

    new_format = "lat_res" in meta
    if new_format:
        # Stored coords are top-left corners; pixel resolution from meta.
        pixel_res_lat = meta["lat_res"]
        pixel_res_lon = meta["lon_res"]
        lat_tl = lat
        lon_tl = lon
    else:
        # Stored coords are patch centers; derive pixel resolution from data.
        pixel_res_lat, pixel_res_lon = _derive_pixel_res(lat, strides[1], lon, strides[2])
        # Convert center → top-left (ESDC grid: lat decreases northward → center is
        # south of top edge by half the block height).
        lat_tl = lat + (block_h / 2) * pixel_res_lat
        lon_tl = lon - (block_w / 2) * pixel_res_lon

    print(f"Pixel resolution: lat={pixel_res_lat:.5f}°  lon={pixel_res_lon:.5f}°  "
          f"({'meta' if new_format else 'derived'})")
    print(f"Patch footprint: {block_h * pixel_res_lat:.2f}° × {block_w * pixel_res_lon:.2f}°")

    # Build world grid at native ESDC resolution (lat descending, lon ascending).
    lat_edges = np.arange(90.0, -90.0, -pixel_res_lat)
    lon_edges = np.arange(-180.0, 180.0, pixel_res_lon)
    n_lat, n_lon = len(lat_edges), len(lon_edges)

    count_grid = np.zeros((n_lat, n_lon), dtype=np.int32)
    for i in range(len(lat_tl)):
        row = int((90.0 - lat_tl[i]) / pixel_res_lat)
        col = int((lon_tl[i] + 180.0) / pixel_res_lon)
        r0 = max(row, 0)
        r1 = min(row + block_h, n_lat)
        c0 = max(col, 0)
        c1 = min(col + block_w, n_lon)
        if r1 > r0 and c1 > c0:
            count_grid[r0:r1, c0:c1] += 1

    covered = np.count_nonzero(count_grid)
    print(f"Grid cells covered: {covered:,} / {n_lat * n_lon:,}  "
          f"(max count = {count_grid.max()})")

    # Mask zeros so they show as the map background.
    display = np.where(count_grid > 0, count_grid, np.nan).astype(float)

    # Coordinate arrays for pcolormesh (cell centers).
    lon_centers = lon_edges + pixel_res_lon / 2
    lat_centers = lat_edges - pixel_res_lat / 2

    world = gpd.read_file(_WORLD_SHP)
    # cmap = plt.cm.YlOrRd.copy()
    cmap = cmocean.cm.dense.copy()
    cmap.set_bad(color="#e8e8e8")

    fig, ax = plt.subplots(figsize=(14, 7))
    world.plot(ax=ax, color="#e8e8e8", edgecolor="none", linewidth=0, zorder=1)

    vmax = args.vmax if args.vmax is not None else count_grid.max()
    norm = mcolors.LogNorm(vmin=1, vmax=vmax)
    im = ax.pcolormesh(
        lon_centers, lat_centers, display,
        cmap=cmap, norm=norm,
        rasterized=True, zorder=2,
    )

    # Country borders on top of the density layer for geographic reference.
    world.plot(ax=ax, color="none", edgecolor="#555555", linewidth=0.4, zorder=3)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.5, aspect=30)
    cbar.set_label("Number of samples covering this pixel", fontsize=10)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(
        f"{args.title} — {len(lat):,} samples (density)",
        fontsize=13,
    )
    ax.set_axis_off()
    fig.tight_layout()

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
