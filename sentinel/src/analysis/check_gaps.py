"""Sanity check: report middle-frame gap fractions and GT validity.

Run from the project root:
    python sentinel/scripts/local/check_gaps.py

Respects S2_DATA_DIR (defaults to ./sentinel/data).
"""
import os
import numpy as np
import rasterio

MIDDLE_T = 2

_data_root = os.environ.get("S2_DATA_DIR", "sentinel/data")

# ── Pre-training ──────────────────────────────────────────────────────────────
ssl_root = os.path.join(_data_root, "SSL-consecutive")
locations = sorted(os.listdir(ssl_root))

print("=== Pre-training: middle-frame gap fraction ===")
print(f"{'loc':>8}  {'mid_nan':>8}  {'<=20%':>6}")
over = 0
for loc in locations:
    loc_dir = os.path.join(ssl_root, loc)
    ts = sorted(d for d in os.listdir(loc_dir) if os.path.isdir(os.path.join(loc_dir, d)))
    if len(ts) < 5:
        continue
    mid = os.path.join(loc_dir, ts[MIDDLE_T])
    bf = sorted(f for f in os.listdir(mid) if f.endswith(".tif"))
    with rasterio.open(os.path.join(mid, bf[0])) as src:
        arr = src.read(1).astype("f4")
    frac = float(np.isnan(arr).mean())
    ok = "YES" if frac <= 0.20 else "NO "
    if frac > 0.20:
        over += 1
    print(f"{loc:>8}  {frac:>8.3f}  {ok:>6}")

print(f"Samples over 20% gap in middle frame: {over}/{len(locations)}")

# ── Gap-filling ───────────────────────────────────────────────────────────────
gf_ssl = os.path.join(_data_root, "gap_filling", "SSL-consecutive")
gf_gt  = os.path.join(_data_root, "gap_filling", "ground_truth")
locs_gf = sorted(os.listdir(gf_ssl))

print()
print("=== Gap-filling: middle NaN | GT NaN | GT clean | gap_mask fraction ===")
print(f"{'loc':>8}  {'mid_nan':>8}  {'gt_nan':>8}  {'gt_clean':>9}  {'mask_frac':>10}")
gt_bad = 0
for loc in locs_gf:
    loc_dir = os.path.join(gf_ssl, loc)
    ts = sorted(d for d in os.listdir(loc_dir) if os.path.isdir(os.path.join(loc_dir, d)))
    if len(ts) < 5:
        continue
    mid = os.path.join(loc_dir, ts[MIDDLE_T])
    bf = sorted(f for f in os.listdir(mid) if f.endswith(".tif"))
    with rasterio.open(os.path.join(mid, bf[0])) as src:
        mid_nan = float(np.isnan(src.read(1).astype("f4")).mean())

    gt_loc = os.path.join(gf_gt, loc)
    gt_ts_dir = os.path.join(gt_loc, sorted(os.listdir(gt_loc))[0])
    gt_bf = sorted(
        f for f in os.listdir(gt_ts_dir)
        if f.endswith(".tif") and f != "gap_mask.tif"
    )
    gt_bands = []
    for f in gt_bf:
        with rasterio.open(os.path.join(gt_ts_dir, f)) as src:
            gt_bands.append(src.read(1).astype("f4"))
    gt_nan = float(np.isnan(np.stack(gt_bands)).mean())
    if gt_nan > 0:
        gt_bad += 1

    with rasterio.open(os.path.join(gt_ts_dir, "gap_mask.tif")) as src:
        mask_frac = float(src.read(1).mean())

    clean = "YES" if gt_nan == 0 else "NO "
    print(f"{loc:>8}  {mid_nan:>8.3f}  {gt_nan:>8.4f}  {clean:>9}  {mask_frac:>10.3f}")

print(f"GT patches with any NaN: {gt_bad}/{len(locs_gf)}")
