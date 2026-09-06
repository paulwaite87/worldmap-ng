#!/usr/bin/env python3
"""Burnable-vegetation mask for the Fire Risk layer (see issue #390).

Fosberg Fire Weather Index (tasks/fire_weather.py) is computed purely from
atmospheric conditions (temperature, humidity, wind), with no concept of whether
there's any vegetation to burn -- open ocean, desert, ice, and bare rock can all
trigger it just as easily as a real forest. This module supplies the other half of
the picture: a boolean "is this cell covered by vegetation that could plausibly
burn" mask, derived from NASA's MODIS Land Cover product (MCD12Q1 v061, IGBP
classification / LC_Type1).

Data source: rather than the raw NASA per-tile MCD12Q1 product (HDF4 format, no
confirmed rasterio/GDAL wheel support for that format here, and would need NASA
Earthdata auth plus a new per-tile mosaic system), this uses a pre-mosaicked
derivative of the same MCD12Q1 v061 IGBP classification published on Zenodo: a
single global Cloud-Optimized GeoTIFF per year, EPSG:4326, ~1km resolution,
CC-BY-SA 4.0 licensed, no authentication required
(https://zenodo.org/records/8367523, concept DOI 10.5281/zenodo.8338927). Verified
live during design, not assumed from documentation: the record's file list uses a
"t1_c_500m_s_{start}_{end}_..." naming convention for the LC_Type1 (IGBP) band, and
Zenodo's "versions/latest" API endpoint always resolves to the record's current
newest published version -- see fetch_latest_zenodo_version()/find_landcover_asset().

IGBP burnable classification (LC_Type1 values 1-17): burnable = forest, shrubland,
savanna, grassland, wetland, and cropland types (1-12, 14) -- wetlands are included
because dried marsh/peat is real, well-documented fire fuel. Not burnable = Urban
(13, fuel-sparse enough that showing risk there is its own false positive),
Permanent Snow/Ice (15), Barren (16), and Water Bodies (17).
"""
import json
import logging
import os
import re

import numpy as np

from atmos_gl.lib.flood_risk import reproject_categorical_max

logger = logging.getLogger(__name__)

# The dataset is only ever accessed as a single whole-file fetch (it's already a
# global mosaic, unlike JRC/MODIS-flood's per-tile products), so this module needs
# no tile-mosaic machinery of its own -- only _reproject_categorical_max's generic
# rasterio reproject(max) mechanics are reused from lib/flood_risk.py, since that's
# equally applicable to any single-band categorical GeoTIFF, not just flood tiles.
_ZENODO_RECORD_ID = "8367523"
ZENODO_VERSIONS_LATEST_URL = (
    f"https://zenodo.org/api/records/{_ZENODO_RECORD_ID}/versions/latest"
)
ZENODO_RECORD_HTML_URL = f"https://zenodo.org/records/{_ZENODO_RECORD_ID}"

# Matches this dataset's own filename convention for the LC_Type1 (IGBP) band, e.g.
# "lc_mcd12q1v061.t1_c_500m_s_20210101_20211231_go_epsg.4326_v20230818.tif" --
# confirmed live against the real Zenodo API response, not assumed.
_T1_FILENAME_RE = re.compile(r"t1_c_500m_s_(\d{8})_(\d{8})_")

# IGBP LC_Type1 values that count as "burnable" for this mask -- see module
# docstring for the reasoning behind wetlands (included) and urban (excluded).
BURNABLE_IGBP_CLASSES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14})

# Confirmed live: this dataset's native resolution is one global ~86400x35849
# mosaic (~3.1 billion pixels) -- reading that wholesale before reprojecting
# OOM-killed the process at ~4GB RSS even for a modest 180x360 destination grid.
# This caps the decimated read (see reproject_categorical_max's max_source_pixels)
# well above any real destination grid's own resolution (a full-globe 0.25deg
# render is ~1.04M points) while keeping the decimated array itself tiny
# (<=25MB as uint8), leaving comfortable headroom for "any burnable nearby"
# accuracy without coming anywhere near the OOM threshold.
_MAX_SOURCE_PIXELS = 25_000_000


def fetch_latest_zenodo_version(timeout: int = 15) -> dict:
    """The current latest-published-version record JSON for this Zenodo dataset.
    Raises on any network/HTTP failure -- callers decide how to log/handle it."""
    import requests

    r = requests.get(ZENODO_VERSIONS_LATEST_URL, timeout=timeout)
    r.raise_for_status()
    return r.json()


def find_landcover_asset(version_json: dict) -> dict | None:
    """Picks the most recent year's LC_Type1 ("t1") band file from a Zenodo
    record's file list (version_json["files"], each a {key, size, checksum, links:
    {self: download_url}} dict -- confirmed live against the real API), or None if
    no matching file is present. Ties/ordering are broken by the end-date embedded
    in the filename, not list order, since the API doesn't guarantee any."""
    candidates = []
    for f in version_json.get("files", []):
        m = _T1_FILENAME_RE.search(f.get("key", ""))
        if m:
            candidates.append((m.group(2), f))
    if not candidates:
        return None
    _end_date, asset = max(candidates, key=lambda c: c[0])
    return asset


def download_landcover_geotiff(download_url: str, dest_path: str) -> None:
    """Fetch the whole GeoTIFF and atomically replace whatever's currently cached
    at dest_path -- mirrors save_jrc_hazard_mosaic's tmp-then-replace pattern."""
    from atmos_gl.lib.gfs import download_whole

    data = download_whole(download_url, timeout=300)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = f"{dest_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, dest_path)


def vegetation_mask_geotiff_cache_path(workdir: str) -> str:
    """Under {workdir}/data (bind-mounted) -- this file is fetched by
    VegetationMaskCollector (running under data_collector) and read by
    FireWeatherUpdater (running under layer_builder), two separate containers, so
    it must live on the directory both share, not a container-local cache dir."""
    return os.path.join(workdir, "data", "vegetation_mask_cache_landcover.tif")


def vegetation_mask_version_cache_path(workdir: str) -> str:
    return os.path.join(workdir, "data", "vegetation_mask_cache_version.json")


def cached_version_id(workdir: str):
    """The Zenodo record id the currently-cached GeoTIFF was fetched from, or None
    if nothing has ever been cached (or the sidecar can't be read)."""
    try:
        with open(vegetation_mask_version_cache_path(workdir)) as f:
            return json.load(f).get("id")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_cached_version_id(workdir: str, version_id) -> None:
    path = vegetation_mask_version_cache_path(workdir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"id": version_id}, f)
    os.replace(tmp_path, path)


def _remap_igbp_to_burnable(source, _src):
    return np.isin(source, list(BURNABLE_IGBP_CLASSES)).astype(np.uint8)


def burnable_vegetation_mask(lat, lon, workdir: str):
    """Boolean burnable-vegetation mask (True where MODIS Land Cover classifies the
    cell as one of BURNABLE_IGBP_CLASSES) sampled at the given lat/lon cell-center
    axes, or None if the land-cover raster hasn't been downloaded yet
    (VegetationMaskCollector's first fetch not yet complete) or fails to read --
    callers fall back to whatever other masking they have, same graceful-degrade
    contract as coastline.py's coastline_land_mask().

    reproject_categorical_max (reused from lib/flood_risk.py) assumes a
    north-first (descending) destination latitude axis, matching every one of its
    existing callers' mosaic grids. ScalarFieldUpdater's LOD-regridded axis
    (regrid_for_lod) is ascending instead, unlike the native fieldstore grid this
    mask is also queried against -- so `lat` may arrive in either order here. This
    flips to descending before calling it and flips the result back to match
    whatever order the caller actually passed in, rather than assuming one.

    Passes max_source_pixels=_MAX_SOURCE_PIXELS -- confirmed live that omitting
    this OOM-kills the process (see _MAX_SOURCE_PIXELS's own comment) against this
    dataset's real, full-resolution global mosaic.
    """
    path = vegetation_mask_geotiff_cache_path(workdir)
    if not os.path.exists(path):
        logger.warning(
            "Vegetation land-cover raster not yet downloaded "
            "(VegetationMaskCollector's first fetch not yet complete); "
            "vegetation mask skipped."
        )
        return None
    try:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        ascending = len(lat) > 1 and lat[0] < lat[-1]
        lat_for_reproject = lat[::-1] if ascending else lat
        mask = reproject_categorical_max(
            path,
            lat_for_reproject,
            lon,
            _remap_igbp_to_burnable,
            max_source_pixels=_MAX_SOURCE_PIXELS,
        )
        if ascending:
            mask = mask[::-1]
        return mask.astype(bool)
    except Exception as exc:  # raster missing/corrupt/unreadable -> graceful fallback
        logger.warning(
            f"Vegetation land-cover raster unavailable ({exc!r}); vegetation mask skipped."
        )
        return None


class VegetationMaskCache:
    """Burnable-vegetation mask, cached per grid for the life of one run -- mirrors
    coastline.py's LandMaskCache exactly, for the same reason (a render task calls
    this repeatedly, once per forecast hour, against the same grid every time).

    Takes `workdir` explicitly (unlike LandMaskCache) because the underlying raster
    is fetched by a separate collector process (VegetationMaskCollector, running
    under data_collector) and consumed here by a render task (running under
    layer_builder) -- workdir is the bind-mounted directory both containers share.

    `key` is an opaque cache key chosen by the caller, not necessarily just the
    array shape: FireWeatherUpdater queries this against two genuinely different
    grids per render (native resolution and LOD-regridded) that can coincidentally
    share a shape, so it includes each grid's latitude endpoints too -- shape alone
    would let one grid's cached mask get silently reused for the other.
    """

    def __init__(self, label: str, workdir: str):
        self._label = label
        self._workdir = workdir
        self._cache = {}

    def get(self, lat, lon, key):
        if key in self._cache:
            return self._cache[key]
        mask = burnable_vegetation_mask(lat, lon, self._workdir)
        self._cache[key] = mask
        if mask is not None:
            logger.info(
                f"{self._label}: built {key} burnable-vegetation mask "
                f"({int(mask.sum())} burnable cells)."
            )
        return mask
