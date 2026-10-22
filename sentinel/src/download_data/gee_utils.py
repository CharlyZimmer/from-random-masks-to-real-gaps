"""Shared GEE utilities for Sentinel-2 spatiotemporal patch downloading.

All reusable functions and classes shared between the pre-training and
gap-filling dataset downloaders.
"""

import json
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from datetime import timedelta
from multiprocessing.dummy import Lock
from typing import Any, Dict, List, Optional, Tuple

import ee
import numpy as np
import rasterio
import shapefile
import urllib3
from googleapiclient.errors import HttpError
from rasterio.transform import Affine
from rtree import index
from shapely.geometry import Point, shape
from torchvision.datasets.utils import download_and_extract_archive

warnings.simplefilter("ignore", UserWarning)


# ---------------------------------------------------------------------------
# Cloud / shadow masking
# ---------------------------------------------------------------------------

def _get_col(col_name: str, aoi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    s2_sr_col = (
        ee.ImageCollection(col_name)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    joined = ee.Join.saveFirst("s2cloudless").apply(
        primary=s2_sr_col,
        secondary=s2_cloudless_col,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined)


def _add_cloud_bands(img: ee.Image, cld_prb_thresh: int) -> ee.Image:
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")
    is_cloud = cld_prb.gt(cld_prb_thresh).rename("clouds")
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def _add_shadow_bands(img: ee.Image, nir_drk_thresh: float, cld_prj_dist: float) -> ee.Image:
    not_water = img.select("SCL").neq(6)
    SR_BAND_SCALE = 1e4
    dark_pixels = (
        img.select("B8")
        .lt(nir_drk_thresh * SR_BAND_SCALE)
        .multiply(not_water)
        .rename("dark_pixels")
    )
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
    cld_proj = (
        img.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, cld_prj_dist * 10)
        .reproject(crs=img.select(0).projection(), scale=100)
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def _add_cld_shdw_mask(
    img: ee.Image,
    cld_prb_thresh: int,
    nir_drk_thresh: float,
    cld_prj_dist: float,
    buffer_m: int,
) -> ee.Image:
    img_cloud = _add_cloud_bands(img, cld_prb_thresh)
    img_cloud_shadow = _add_shadow_bands(img_cloud, nir_drk_thresh, cld_prj_dist)
    is_cld_shdw = img_cloud_shadow.select("clouds").add(img_cloud_shadow.select("shadows")).gt(0)
    is_cld_shdw = (
        is_cld_shdw.focalMin(2)
        .focalMax(buffer_m * 2 / 20)
        .reproject(crs=img.select([0]).projection(), scale=20)
        .rename("cloudmask")
    )
    return img_cloud_shadow.addBands(is_cld_shdw)


def _apply_cld_shdw_mask(img: ee.Image) -> ee.Image:
    """Mask optical bands using cloud/shadow mask but keep all non-optical bands."""
    not_cld_shdw = img.select("cloudmask").Not()
    band_names = img.bandNames()

    def is_optical(b):
        return ee.Algorithms.If(ee.String(b).slice(0, 1).equals("B"), b, None)

    optical_bands = band_names.map(is_optical).removeAll([None])
    img_wo_optical = img.select(band_names.removeAll(optical_bands))
    masked_optical = img.select("B.*").updateMask(not_cld_shdw)
    return img_wo_optical.addBands(masked_optical)


def _dedupe_by_timestamp(col: ee.ImageCollection) -> ee.ImageCollection:
    """Remove duplicate overlapping tiles for the same UTC date (keep one per day)."""

    def add_date_key(img):
        day = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        return img.set("date_key", day)

    col = col.map(add_date_key)
    join = ee.Join.saveAll(matchesKey="matches")
    joined = ee.ImageCollection(
        join.apply(primary=col, secondary=col, condition=ee.Filter.equals(leftField="date_key", rightField="date_key"))
    )

    def pick_one(img):
        matches = ee.ImageCollection.fromImages(ee.List(img.get("matches"))).sort("system:index")
        chosen = ee.Image(matches.first())
        return chosen.set("date_key", img.get("date_key"))

    return joined.map(pick_one).distinct("date_key")


def _has_valid_middle_patch(
    image_list: ee.List,
    aoi: ee.Geometry,
    radius_m: float,
    max_gap_ratio: float,
) -> ee.Number:
    region = aoi.buffer(radius_m).bounds()
    middle_img = ee.Image(image_list.get(2))

    # After _apply_cld_shdw_mask the optical bands are masked (mask=0) for
    # both cloud/shadow pixels AND out-of-swath pixels.  unmask(0) makes
    # reduceRegion count all region pixels in the denominator so the gap
    # fraction includes boundary pixels, not just in-swath cloud cover.
    valid_mask = middle_img.select("B2").mask().unmask(0)
    valid_fraction = valid_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
    ).get("B2")

    gap_fraction = ee.Number(1).subtract(ee.Number(valid_fraction))
    return gap_fraction.lte(max_gap_ratio)


def _getinfo_with_retries(obj: Any, retries: int = 3, sleep_s: float = 2.0) -> Any:
    """Client-side getInfo() with exponential backoff."""
    for k in range(retries):
        try:
            return obj.getInfo()
        except (ee.EEException, HttpError) as e:
            delay = (sleep_s * (2 ** k)) + float(np.random.uniform(0, 0.5))
            print(f"Error fetching info (attempt {k + 1}/{retries}): {e}")
            time.sleep(delay)
    return None


def _build_time_series_patches(
    coords: List[float],
    start_date: str,
    end_date: str,
    bands: List[str],
    radius_m: float,
    n_timesteps: int,
    max_gap_ratio: float,
    cld_prb_thresh: int,
    nir_drk_thresh: float,
    cld_prj_dist: float,
    buffer_m: int,
    col_name: str,
    patch_size: Optional[int] = None,
    dtype: str = "float32",
) -> Tuple[Optional[List[Dict[str, Any]]], List[float]]:
    """Sample a time series around coords using cloud masking logic.

    Earth Engine's sampleRectangle() does not guarantee a fixed output raster size
    for a given geometry. If patch_size is provided, this function center-crops
    every returned band to (patch_size, patch_size) and adjusts the returned
    geospatial coords accordingly.
    """
    aoi = ee.Geometry.Point(coords[0], coords[1])
    col = _get_col(col_name, aoi, start_date, end_date)
    col = col.map(
        lambda img: _add_cld_shdw_mask(
            img,
            cld_prb_thresh=cld_prb_thresh,
            nir_drk_thresh=nir_drk_thresh,
            cld_prj_dist=cld_prj_dist,
            buffer_m=buffer_m,
        )
    ).map(_apply_cld_shdw_mask)
    col = _dedupe_by_timestamp(col)

    images = col.sort("system:time_start").toList(col.size())
    n_images = images.size()

    valid_indices = ee.List.sequence(0, n_images.subtract(n_timesteps))

    def check_valid(idx):
        idx = ee.Number(idx)
        window = images.slice(idx, idx.add(n_timesteps))
        return ee.Algorithms.If(
            _has_valid_middle_patch(window, aoi, radius_m, max_gap_ratio), idx, None
        )

    valid_samples = valid_indices.map(check_valid).removeAll([None])
    valid_list = _getinfo_with_retries(valid_samples, retries=2, sleep_s=1.0)

    if not valid_list:
        return None, coords

    # Take first valid window
    start_idx = int(valid_list[0])
    window = images.slice(start_idx, start_idx + n_timesteps)

    region = ee.Geometry.Point(coords[0], coords[1]).buffer(radius_m).bounds()

    by_timestep: Dict[int, Dict[str, np.ndarray]] = {t: {} for t in range(n_timesteps)}
    old_size: Optional[Tuple[int, int]] = None
    raw_coords: Optional[List[List[float]]] = None

    for t in range(n_timesteps):
        img = ee.Image(window.get(t))
        # Reproject to B2's 10 m projection so sampleRectangle works with
        # mixed-resolution bands (B5/B6/B7/B8A/B11/B12 are native 20 m).
        img_t = img.select(bands).reproject(crs=img.select("B2").projection())
        patch_t = img_t.sampleRectangle(region, defaultValue=0)
        features_t = _getinfo_with_retries(patch_t, retries=6, sleep_s=1.0)
        if features_t is None:
            return None, coords

        if raw_coords is None:
            coords0 = np.array(features_t["geometry"]["coordinates"][0])
            raw_coords = [
                [coords0[:, 0].min(), coords0[:, 1].max()],
                [coords0[:, 0].max(), coords0[:, 1].min()],
            ]

        for b in bands:
            if b not in features_t["properties"]:
                return None, coords
            arr = np.atleast_3d(features_t["properties"][b]).astype(dtype)
            if old_size is None:
                old_size = (int(arr.shape[0]), int(arr.shape[1]))
            by_timestep[t][b] = arr

    if old_size is None or raw_coords is None:
        return None, coords

    patch_coords = raw_coords
    if patch_size is not None:
        if old_size[0] < patch_size or old_size[1] < patch_size:
            return None, coords
        patch_coords = adjust_coords(raw_coords, old_size, (patch_size, patch_size))

    window_imgs = ee.ImageCollection.fromImages(window)
    window_meta = _getinfo_with_retries(
        window_imgs.aggregate_array("system:index"), retries=2, sleep_s=1.0
    )
    window_times = _getinfo_with_retries(
        window_imgs.aggregate_array("system:time_start"), retries=2, sleep_s=1.0
    )
    if window_meta is None or window_times is None:
        return None, coords

    patches: List[Dict[str, Any]] = []
    for t in range(n_timesteps):
        raster = OrderedDict()
        for b in bands:
            if b not in by_timestep[t]:
                continue
            img_arr = by_timestep[t][b]
            if patch_size is not None:
                img_arr = center_crop(img_arr, out_size=(patch_size, patch_size))
            raster[b] = img_arr

        metadata = {
            "properties": {
                "system:index": window_meta[t],
                "system:time_start": window_times[t],
            }
        }
        if raster:
            band_stack = np.concatenate(list(raster.values()), axis=-1)  # (H, W, C)
            gap_mask = np.all(band_stack == 0, axis=-1)                   # (H, W)
            for arr in raster.values():
                arr[gap_mask] = np.nan

        patches.append(
            OrderedDict(
                {
                    "raster": raster,
                    "coords": patch_coords,
                    "metadata": metadata,
                }
            )
        )

    return patches, coords


_DONOR_SEED = 42


def _get_donor_gap_map(
    coords: List[float],
    start_date: str,
    end_date: str,
    exclude_system_indexes: List[str],
    bands: List[str],
    radius_m: float,
    patch_size: Optional[int],
    cld_prb_thresh: int,
    nir_drk_thresh: float,
    cld_prj_dist: float,
    buffer_m: int,
    col_name: str,
    min_gap_ratio: float = 0.05,
    max_gap_ratio: float = 0.85,
    dtype: str = "float32",
    window_time_starts: Optional[List[int]] = None,
    donor_probe_k: Optional[int] = None,
    idx: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Find a donor image in the same time window and extract its binary gap mask.

    The donor is chosen from images in [start_date, end_date] that are NOT among
    the 5 selected timesteps (identified by their GEE system:index). Candidates are
    filtered to have a cloud/gap fraction in [min_gap_ratio, max_gap_ratio] to ensure
    the synthetic gap is both meaningful and not completely opaque. One candidate is
    selected uniformly at random to achieve dataset-level diversity. When `idx` is
    given, this selection is drawn from a locally-seeded generator (deterministic
    per idx) instead of the shared global `random` module, so the choice is
    invariant to worker thread interleaving and reruns -- both downloader scripts
    call this from a `multiprocessing.dummy.Pool` (thread pool sharing one
    process), which would otherwise race on the global RNG.

    `window_time_starts` (the reused 5-step window's system:time_start values, ms)
    and `donor_probe_k` bound the server-side cost when [start_date, end_date] is
    wide: only the `donor_probe_k` donor acquisitions closest in time to that
    window's middle frame have their gap fraction probed, so the (expensive)
    per-candidate reduceRegion count stays fixed and the donor stays in the same
    season as the target frame. Both default to None (probe every candidate),
    which is what the eval-set downloader uses.

    If the uniformly-chosen donor's raster download fails (missing band, undersized
    raster, or a GEE error that survives retries), the remaining candidates are
    tried in ascending-time order, up to a small bounded number of attempts,
    instead of giving up on the sample.

    Returns:
        (gap_mask, gap_fraction):
            gap_mask  — bool (H, W) array; True where the synthetic gap should be.
            gap_fraction — cloud fraction of the selected donor (0.0 if no donor found).
        Returns (None, 0.0) if no suitable donor is found.
    """
    aoi = ee.Geometry.Point(coords[0], coords[1])
    region = aoi.buffer(radius_m).bounds()

    # Rebuild the same collection with cloud/shadow masking
    col = _get_col(col_name, aoi, start_date, end_date)
    col = col.map(
        lambda img: _add_cld_shdw_mask(
            img,
            cld_prb_thresh=cld_prb_thresh,
            nir_drk_thresh=nir_drk_thresh,
            cld_prj_dist=cld_prj_dist,
            buffer_m=buffer_m,
        )
    ).map(_apply_cld_shdw_mask)
    col = _dedupe_by_timestamp(col)

    # Remove the 5 selected window images from the donor pool
    col = col.filter(
        ee.Filter.inList("system:index", ee.List(exclude_system_indexes)).Not()
    )

    n_donors = _getinfo_with_retries(col.size(), retries=2, sleep_s=1.0)
    if not n_donors:
        return None, 0.0

    sorted_col = col.sort("system:time_start")
    images = sorted_col.toList(n_donors)

    # Restrict the (expensive) per-candidate gap-fraction probe to the K donor
    # acquisitions closest in time to the reused 5-step window's middle frame, so
    # the server-side reduceRegion cost stays bounded no matter how wide
    # [start_date, end_date] is and the donor stays in the same season as the
    # target frame. probe_positions holds absolute positions into `images`.
    probe_positions = list(range(n_donors))
    if window_time_starts and donor_probe_k is not None and n_donors > donor_probe_k:
        ref_t = sorted(window_time_starts)[len(window_time_starts) // 2]
        donor_times = _getinfo_with_retries(
            sorted_col.aggregate_array("system:time_start"), retries=2, sleep_s=1.0
        )
        if donor_times and len(donor_times) == n_donors:
            probe_positions = sorted(
                range(n_donors),
                key=lambda i: (abs(int(donor_times[i]) - int(ref_t)), i),
            )[:donor_probe_k]
            probe_positions.sort()

    # Compute total gap fraction (cloud/shadow + out-of-swath) for each probed
    # candidate. Same approach as _has_valid_middle_patch: use the optical band
    # mask so that boundary pixels are counted in the denominator.
    def get_gap_frac(pos: Any) -> Any:
        img = ee.Image(images.get(ee.Number(pos).int()))
        valid_mask = img.select("B2").mask().unmask(0)
        result = valid_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=20,
            maxPixels=1e8,
        )
        return ee.Number(1).subtract(ee.Number(result.get("B2")))

    fracs_ee = ee.List(probe_positions).map(get_gap_frac)
    gap_fractions = _getinfo_with_retries(fracs_ee, retries=2, sleep_s=1.0)

    if gap_fractions is None:
        return None, 0.0

    # Pair each probed gap fraction back with its absolute position in `images`.
    probed = [
        (probe_positions[j], gf)
        for j, gf in enumerate(gap_fractions)
        if gf is not None
    ]

    # Select candidates within the acceptable gap fraction range
    candidates = [
        (pos, gf) for pos, gf in probed
        if min_gap_ratio <= float(gf) <= max_gap_ratio
    ]

    if not candidates:
        # Widen range to ensure at least some synthetic gap is introduced
        candidates = [
            (pos, gf) for pos, gf in probed
            if 0.01 <= float(gf) <= 0.99
        ]

    if not candidates:
        return None, 0.0

    # Deterministic per-idx try order. try_order[0] is the seeded random.choice
    # pick (the same selection for a given seed + candidate list); the remaining
    # candidates, in ascending-time order, are only the fallback
    # order used when a donor's raster download fails.
    rng = random.Random(f"{_DONOR_SEED}:{idx}") if idx is not None else random
    primary = rng.choice(candidates)
    try_order = [primary] + [c for c in candidates if c != primary]

    max_attempts = 5
    for selected_pos, selected_gf in try_order[:max_attempts]:
        donor_img = ee.Image(images.get(selected_pos))
        # Reproject to B2's 10 m projection to handle mixed native resolutions (10 m / 20 m).
        donor_img_t = donor_img.select(bands).reproject(crs=donor_img.select("B2").projection())
        patch = donor_img_t.sampleRectangle(region, defaultValue=0)
        features = _getinfo_with_retries(patch, retries=6, sleep_s=1.0)

        if features is None:
            continue

        # Recover per-band arrays
        band_arrays: List[np.ndarray] = []
        old_size: Optional[Tuple[int, int]] = None
        missing_band = False
        for b in bands:
            if b not in features["properties"]:
                missing_band = True
                break
            arr = np.atleast_3d(features["properties"][b]).astype(dtype)  # (H, W, 1)
            if old_size is None:
                old_size = (int(arr.shape[0]), int(arr.shape[1]))
            band_arrays.append(arr)
        if missing_band or old_size is None:
            continue

        # Enforce the same patch size as the main time series
        if patch_size is not None:
            if old_size[0] < patch_size or old_size[1] < patch_size:
                continue
            band_arrays = [center_crop(a, (patch_size, patch_size)) for a in band_arrays]

        # Build binary gap mask: pixels where all bands are 0 (cloud-masked by GEE)
        band_stack = np.concatenate(band_arrays, axis=-1)  # (H, W, C)
        gap_mask = np.all(band_stack == 0, axis=-1)        # (H, W) bool

        return gap_mask, float(selected_gf)

    return None, 0.0


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class OverlapError(Exception):
    pass


class UniformSampler:
    def sample_point(self) -> List[float]:
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-90, 90)
        return [lon, lat]


class GaussianSampler:
    def __init__(
        self,
        interest_points: Optional[List[List[float]]] = None,
        num_cities: int = 1000,
        std: float = 20,
    ) -> None:
        if interest_points is None:
            cities = self.get_world_cities("sentinel/data/world_cities")
            self.interest_points = self.get_interest_points(cities, size=num_cities)
        else:
            self.interest_points = interest_points
        self.std = std

    def sample_point(self) -> List[float]:
        rng = np.random.default_rng()
        point = rng.choice(self.interest_points)
        std = self.km2deg(self.std)
        lon, lat = np.random.normal(loc=point, scale=[std, std])
        return [lon, lat]

    @staticmethod
    def get_world_cities(download_root: str = "world_cities") -> List[Dict[str, Any]]:
        url = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip"  # noqa: E501
        filename = "worldcities.csv"
        if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
            download_and_extract_archive(url, download_root)
        import csv
        with open(os.path.join(download_root, filename), encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
            cities = []
            for row in reader:
                row["population"] = (
                    row["population"].replace(".", "") if row["population"] else "0"
                )
                cities.append(row)
        return cities

    @staticmethod
    def get_interest_points(
        cities: List[Dict[str, str]], size: int = 10000
    ) -> List[List[float]]:
        cities = sorted(cities, key=lambda c: int(c["population"]), reverse=True)[:size]
        points = [[float(c["lng"]), float(c["lat"])] for c in cities]
        return points

    @staticmethod
    def km2deg(kms: float, radius: float = 6371) -> float:
        return kms / (2.0 * radius * np.pi / 360.0)

    @staticmethod
    def deg2km(deg: float, radius: float = 6371) -> float:
        return deg * (2.0 * radius * np.pi / 360.0)


class BoundedUniformSampler:
    def __init__(self, boundaries: Any = None) -> None:
        if boundaries is None:
            self.boundaries = self.get_country_boundaries()
        else:
            self.boundaries = boundaries

    def sample_point(self) -> List[float]:
        minx, miny, maxx, maxy = self.boundaries.bounds
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        p = Point(lon, lat)
        if self.boundaries.contains(p):
            return [p.x, p.y]
        else:
            return self.sample_point()

    @staticmethod
    def get_country_boundaries(
        download_root: str = os.path.expanduser("~/.cache/naturalearth"),
    ) -> Any:
        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"  # noqa: E501
        filename = "ne_110m_admin_0_countries.shp"
        if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
            download_and_extract_archive(url, download_root)
        sf = shapefile.Reader(os.path.join(download_root, filename))
        return shape(sf.shapes().__geo_interface__)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def date2str(d: Any) -> str:
    return d.strftime("%Y-%m-%d")


def get_period(d: Any, days: int = 5) -> Tuple[str, str, str, str]:
    date1 = d - timedelta(days=days / 2)
    date2 = d + timedelta(days=days / 2)
    date3 = date1 - timedelta(days=365)
    date4 = date2 - timedelta(days=365)
    return (
        date2str(date1),
        date2str(date2),
        date2str(date3),
        date2str(date4),
    )


# ---------------------------------------------------------------------------
# Legacy collection helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def maskS2clouds(args: Any, image: ee.Image) -> ee.Image:
    qa = image.select(args.qa_band)
    cloudBitMask = 1 << args.qa_cloud_bit
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    return image.updateMask(mask)


def get_collection(
    collection_name: str, meta_cloud_name: str, cloud_pct: float
) -> ee.ImageCollection:
    collection = ee.ImageCollection(collection_name)
    collection = collection.filter(ee.Filter.lt(meta_cloud_name, cloud_pct))
    return collection


def filter_collection(
    collection: ee.ImageCollection,
    coords: List[float],
    period: Tuple[str, str, str, str],
) -> ee.ImageCollection:
    filtered = collection
    if period is not None:
        filtered = filtered.filter(
            ee.Filter.Or(
                ee.Filter.date(period[0], period[1]),
                ee.Filter.date(period[2], period[3]),
            )
        )
    filtered = filtered.filterBounds(ee.Geometry.Point(coords))
    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f"ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) "
            f"between {period[0]} and {period[1]}."
        )
    return filtered


def get_properties(image: ee.Image) -> Any:
    return image.getInfo()


# ---------------------------------------------------------------------------
# Spatial utilities
# ---------------------------------------------------------------------------

def center_crop(
    img: "np.ndarray[Any, np.dtype[Any]]", out_size: Tuple[int, int]
) -> "np.ndarray[Any, np.dtype[Any]]":
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = (image_height - crop_height + 1) // 2
    crop_left = (image_width - crop_width + 1) // 2
    return img[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]


def adjust_coords(
    coords: List[List[float]], old_size: Tuple[int, int], new_size: Tuple[int, int]
) -> List[List[float]]:
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [
            coords[0][0] + ((xoff + new_size[1]) * xres),
            coords[0][1] - ((yoff + new_size[0]) * yres),
        ],
    ]


# ---------------------------------------------------------------------------
# Legacy patch download functions (kept for backward compatibility)
# ---------------------------------------------------------------------------

def get_patch(
    collection: ee.ImageCollection,
    center_coord: List[float],
    radius: float,
    bands: List[str],
    crop: Optional[Dict[str, Any]] = None,
    dtype: str = "float32",
) -> Dict[str, Any]:
    image = collection.sort("system:time_start", False).first()
    region = ee.Geometry.Point(center_coord).buffer(radius).bounds()
    patch = image.select(*bands).sampleRectangle(region, defaultValue=0)

    features = patch.getInfo()

    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features["properties"][band])
        if crop is not None:
            img = center_crop(img, out_size=crop[band])
        raster[band] = img.astype(dtype)

    coords0 = np.array(features["geometry"]["coordinates"][0])
    coords = [
        [coords0[:, 0].min(), coords0[:, 1].max()],
        [coords0[:, 0].max(), coords0[:, 1].min()],
    ]
    if crop is not None:
        band = bands[0]
        old_size = (
            len(features["properties"][band]),
            len(features["properties"][band][0]),
        )
        new_size = raster[band].shape[:2]
        coords = adjust_coords(coords, old_size, new_size)

    return OrderedDict(
        {"raster": raster, "coords": coords, "metadata": get_properties(image)}
    )


def get_random_patches_match(
    idx: int,
    collection: ee.ImageCollection,
    bands: List[str],
    crops: Dict[str, Any],
    dtype: str,
    dates: List[Any],
    radius: float,
    debug: bool = False,
    match_coords: Dict[str, Any] = {},
) -> Tuple[Optional[List[Dict[str, Any]]], List[float]]:
    coords = match_coords[str(idx)]
    periods = [get_period(d, days=60) for d in dates]

    try:
        filtered_collections = [filter_collection(collection, coords, p) for p in periods]
        patches = [
            get_patch(c, coords, radius, bands=bands, crop=crops, dtype=dtype)
            for c in filtered_collections
        ]
    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        return None, coords

    return patches, coords


def get_random_patches_rtree(
    idx: int,
    collection: ee.ImageCollection,
    bands: List[str],
    crops: Dict[str, Any],
    dtype: str,
    sampler: GaussianSampler,
    dates: List[Any],
    radius: float,
    debug: bool = False,
    rtree_obj: Optional[index.Index] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    coords = sampler.sample_point()
    new_coord = (coords[0], coords[1])

    if rtree_obj is None:
        raise ValueError("rtree_obj must be provided when overlap_check='rtree'")

    try:
        for i in rtree_obj.nearest(new_coord, num_results=1, objects=True):
            distance = np.sqrt(
                sampler.deg2km(abs(new_coord[0] - i.bbox[2])) ** 2
                + sampler.deg2km(abs(new_coord[1] - i.bbox[3])) ** 2
            )
            if distance < (1.5 * radius / 1000):
                raise OverlapError
        rtree_obj.insert(
            len(rtree_obj) - 1, (new_coord[0], new_coord[1], new_coord[0], new_coord[1])
        )
    except OverlapError:
        patches, center_coord = get_random_patches_rtree(
            idx, collection, bands, crops, dtype, sampler, dates, radius, debug, rtree_obj
        )

    periods = [get_period(d, days=60) for d in dates]

    try:
        filtered_collections = [filter_collection(collection, coords, p) for p in periods]
        patches = [
            get_patch(c, coords, radius, bands=bands, crop=crops, dtype=dtype)
            for c in filtered_collections
        ]
        center_coord = coords
    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        rtree_obj.insert(
            len(rtree_obj) - 1, (new_coord[0], new_coord[1], new_coord[0], new_coord[1])
        )
        patches, center_coord = get_random_patches_rtree(
            idx, collection, bands, crops, dtype, sampler, dates, radius, debug, rtree_obj
        )

    return patches, center_coord


def get_random_patches_grid(
    idx: int,
    collection: ee.ImageCollection,
    bands: List[str],
    crops: Dict[str, Any],
    dtype: str,
    sampler: GaussianSampler,
    dates: List[Any],
    radius: float,
    debug: bool = False,
    grid_dict: Dict[Tuple[int, int], Any] = {},
) -> Tuple[List[Dict[str, Any]], List[float]]:
    coords = sampler.sample_point()

    try:
        new_coord = (coords[0], coords[1])
        gridIndex = (math.floor(new_coord[0] + 180), math.floor(new_coord[1] + 90))

        if gridIndex not in grid_dict.keys():
            grid_dict[gridIndex] = {new_coord}
        else:
            for coord in grid_dict[gridIndex]:
                distance = np.sqrt(
                    sampler.deg2km(abs(new_coord[0] - coord[0])) ** 2
                    + sampler.deg2km(abs(new_coord[1] - coord[1])) ** 2
                )
                if distance < (1.5 * radius / 1000):
                    raise OverlapError
            grid_dict[gridIndex].add(new_coord)
    except OverlapError:
        patches, center_coord = get_random_patches_grid(
            idx, collection, bands, crops, dtype, sampler, dates, radius, debug,
            grid_dict=grid_dict
        )

    periods = [get_period(d, days=30) for d in dates]

    try:
        filtered_collections = [filter_collection(collection, coords, p) for p in periods]
        patches = [
            get_patch(c, coords, radius, bands=bands, crop=crops, dtype=dtype)
            for c in filtered_collections
        ]
        center_coord = coords
    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        patches, center_coord = get_random_patches_grid(
            idx, collection, bands, crops, dtype, sampler, dates, radius, debug,
            grid_dict=grid_dict
        )

    return patches, center_coord


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def save_geotiff(
    img: "np.ndarray[Any, np.dtype[Any]]", coords: List[List[float]], filename: str
) -> None:
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(
        coords[0][0] - xres / 2, coords[0][1] + yres / 2
    ) * Affine.scale(xres, -yres)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": channels,
        "crs": "+proj=latlong",
        "transform": transform,
        "dtype": img.dtype,
        "compress": "lzw",
        "predictor": 2,
    }
    with rasterio.open(filename, "w", **profile) as f:
        f.write(img.transpose(2, 0, 1))


def save_patch(
    raster: Dict[str, Any],
    coords: List[List[float]],
    metadata: Dict[str, Any],
    path: str,
) -> None:
    patch_id = metadata["properties"]["system:index"]
    patch_path = os.path.join(path, patch_id)
    os.makedirs(patch_path, exist_ok=True)

    band_names = list(raster.keys())
    band_stack = np.concatenate(list(raster.values()), axis=-1)  # (H, W, C)
    save_geotiff(band_stack, coords, os.path.join(patch_path, "S2C.tif"))

    metadata_to_save = {
        "properties": metadata["properties"],
        "coords": coords,
        "bands": band_names,
    }
    with open(os.path.join(patch_path, "metadata.json"), "w") as f:
        json.dump(metadata_to_save, f)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

class Counter:
    def __init__(self, start: int = 0) -> None:
        self.value = start
        self.lock = Lock()

    def update(self, delta: int = 1) -> int:
        with self.lock:
            self.value += delta
            return self.value


def fix_random_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
