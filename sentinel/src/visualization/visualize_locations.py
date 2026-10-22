"""Save a global map of Sentinel sample locations to img/sentinel_locations.png."""
import argparse
import glob
import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pyogrio

_DEFAULT_DATA_DIR = os.environ.get("S2_DATA_DIR", "sentinel/data")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
OUT_PATH = os.path.join(_REPO_ROOT, "img", "sentinel_locations.png")

_WORLD_SHP = os.path.join(
    os.path.dirname(pyogrio.__file__),
    "tests", "fixtures", "naturalearth_lowres", "naturalearth_lowres.shp",
)


def _load_from_location_json(ssl_dir: str) -> pd.DataFrame:
    records = []
    for path in sorted(glob.glob(os.path.join(ssl_dir, "*", "location.json"))):
        with open(path) as f:
            records.append(json.load(f))
    return pd.DataFrame(records, columns=["idx", "lon", "lat"]) if records else pd.DataFrame()


def _load_from_checked_locations(ssl_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(ssl_dir), "checked_locations.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, header=None, names=["idx", "lon", "lat", "success_flag"])
    return df[df["success_flag"] > 0][["idx", "lon", "lat"]].reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="Data root directory (SSL-consecutive/ is expected inside).")
    p.add_argument("--out", default=OUT_PATH, help="Output PNG path.")
    p.add_argument("--title", default="Sentinel Pre-Training Dataset", help="Dataset label used in the plot title.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ssl_dir = os.path.join(args.data_dir, "SSL-consecutive")
    df = _load_from_location_json(ssl_dir)
    source = "location.json files"
    if df.empty:
        df = _load_from_checked_locations(ssl_dir)
        source = "checked_locations.csv"

    if df.empty:
        raise RuntimeError(f"No location data found under {ssl_dir}")

    print(f"Loaded {len(df):,} locations from {source}")

    world = gpd.read_file(_WORLD_SHP)

    fig, ax = plt.subplots(figsize=(14, 7))
    world.plot(ax=ax, color="#e8e8e8", edgecolor="#888888", linewidth=0.4)
    ax.scatter(df["lon"], df["lat"], s=6, c="red", alpha=0.5, linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(f"{args.title} — {len(df):,} locations", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
