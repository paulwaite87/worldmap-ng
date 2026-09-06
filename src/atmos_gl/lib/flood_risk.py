#!/usr/bin/env python3
"""Shared helpers for the flood_risk layer, used by both collectors
(collectors/flood_risk.py) and the updater (tasks/flood_risk.py) -- same "one source
of truth for path/URL/math conventions" role lib/greenhouse_gases.py plays for the
greenhouse_gases layer.

Covers: JRC Global River Flood Hazard Maps' tile index/download/mosaic conventions
(Historical mode) and NASA LANCE MODIS flood product's tile index/download/mosaic
conventions (Live mode, "Observed Current Inundation" -- see collectors/flood_risk.py's
module docstring for why this replaced the original GloFAS ensemble-discharge-forecast
design). Both modes share the same underlying shape -- a small-integer categorical
raster tiled in 10x10deg blocks, mosaicked onto one shared global grid via
_reproject_categorical_max -- despite being unrelated data sources.
"""
import json
import logging
import os
import re
import time

import numpy as np

logger = logging.getLogger(__name__)


def _flood_risk_cache_dir() -> str:
    """Container-local cache dir for per-tile downloads -- NOT bind-mounted, same
    "ephemeral across container recreation, persistent across worker restarts"
    convention as coastline.py's _gshhg_cache_dir()."""
    return os.path.join(os.path.expanduser("~"), ".local", "share", "flood_risk")


# --- JRC Global River Flood Hazard Maps (Historical mode) ------------------------
#
# Open FTP, no auth, CC-BY-4.0 (confirmed live during issue #371's spike; the
# dataset's own catalogue page is WMS-only/gated, a red herring -- this FTP tree is
# the real open-data distribution point). 271 tiles globally (10x10deg WGS84
# blocks, see tile_extents.geojson), 3 arc-second (~90m) native resolution, two file
# variants per tile per return period (_depth.tif raw metres, _depth_reclass.tif
# categorical: 1:<1m, 2:1-3m, 3:3-10m, 4:>10m, nodata=255) -- only the reclass
# variant is fetched (~515MB total for RP100 across all 271 tiles), since it's
# already exactly the categorical severity classification this layer displays,
# confirmed by the dataset's own README as "consistent with the GloFAS 'Flood
# hazard 100-year return period' static layer."
# Public (no leading underscore): also used by FloodRiskHistoricalCollector.source_url()
# for the Data Status page's clickable-label link, same "hardcoded source, no
# data_collector.datasources entry" convention as StormsCollector's own URLs.
JRC_BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
# 100-year return period (see issue #371's Implementation Decisions).
_JRC_RETURN_PERIOD = "RP100"
_JRC_TILE_EXTENTS_URL = f"{JRC_BASE_URL}/tile_extents.geojson"

# Working resolution of the cached global mosaics -- shared by BOTH modes (JRC's
# native ~90m and MODIS's native 250m are each far finer than a global view needs).
# Originally chosen to match GloFAS forecast's own 0.05deg operational grid; kept
# unchanged now that GloFAS is gone since it's still a reasonable global working
# resolution and changing it would invalidate every cached mosaic for no benefit.
JRC_MOSAIC_GRID_STEP_DEG = 0.05


def _jrc_cache_dir() -> str:
    return os.path.join(_flood_risk_cache_dir(), "jrc")


def _jrc_tiles_cache_dir() -> str:
    return os.path.join(_jrc_cache_dir(), _JRC_RETURN_PERIOD)


def jrc_tile_extents_cache_path() -> str:
    return os.path.join(_jrc_cache_dir(), "tile_extents.geojson")


def jrc_tile_cache_path(tile_id: int, tile_name: str) -> str:
    return os.path.join(
        _jrc_tiles_cache_dir(), f"ID{tile_id}_{tile_name}_{_JRC_RETURN_PERIOD}_depth_reclass.tif"
    )


def jrc_tile_download_url(tile_id: int, tile_name: str) -> str:
    return (
        f"{JRC_BASE_URL}/{_JRC_RETURN_PERIOD}/"
        f"ID{tile_id}_{tile_name}_{_JRC_RETURN_PERIOD}_depth_reclass.tif"
    )


def jrc_hazard_mosaic_cache_path(workdir: str) -> str:
    """Cache path for the final assembled global mosaic -- under {workdir}/data
    (bind-mounted, survives container recreation), unlike the raw per-tile downloads
    and tile index below (home-dir cache, GSHHG's own convention): the 271-tile
    download is the expensive one-time cost worth surviving a container rebuild for."""
    return os.path.join(workdir, "data", "flood_risk_cache_jrc_hazard_mosaic.nc")


def ensure_jrc_tile_extents_cached() -> str:
    """Download + cache the 271-tile index (tiny, ~100KB), if not already cached.
    Returns the cached .geojson path. Raises on failure -- same graceful-fallback
    contract as ensure_jrc_tile_cached below."""
    dest = jrc_tile_extents_cache_path()
    if os.path.exists(dest):
        return dest

    from atmos_gl.lib.gfs import download_whole

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Flood Risk: downloading JRC tile index (one-time)...")
    data = download_whole(_JRC_TILE_EXTENTS_URL, timeout=60)

    tmp_dest = f"{dest}.tmp"
    with open(tmp_dest, "wb") as f:
        f.write(data)

    try:
        with open(tmp_dest) as f:
            parsed = json.load(f)
        if not parsed.get("features"):
            raise ValueError("tile_extents.geojson has no features")
    except Exception:
        os.remove(tmp_dest)
        raise

    os.replace(tmp_dest, dest)
    logger.info(f"Flood Risk: cached JRC tile index -> {dest}")
    return dest


def load_jrc_tile_index(path: str) -> list[dict]:
    """List of {"id", "name", "bounds"} from a cached tile_extents.geojson --
    `bounds` is (lon_min, lat_min, lon_max, lat_max), derived from each feature's
    Polygon ring rather than trusted as pre-sorted corners."""
    with open(path) as f:
        parsed = json.load(f)

    tiles = []
    for feature in parsed["features"]:
        coords = feature["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        tiles.append(
            {
                "id": feature["properties"]["id"],
                "name": feature["properties"]["name"],
                "bounds": (min(lons), min(lats), max(lons), max(lats)),
            }
        )
    return tiles


def ensure_jrc_tile_cached(tile_id: int, tile_name: str) -> str:
    """Download + cache one JRC reclass tile (~0.1-8.5MB), if not already cached.
    Returns the cached .tif path. Raises on failure -- callers skip this tile for
    the current cycle and retry next time; the already-cached tiles from a prior
    partial pass are untouched, so a multi-cycle download naturally resumes rather
    than restarting.

    Validated by opening with rasterio and reading a single pixel before being
    trusted as cached -- confirmed live that plain downloads against JRC's host
    CAN be interrupted mid-transfer for a 271-tile batch."""
    dest = jrc_tile_cache_path(tile_id, tile_name)
    if os.path.exists(dest):
        return dest

    from atmos_gl.lib.gfs import download_whole

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = jrc_tile_download_url(tile_id, tile_name)
    data = download_whole(url, timeout=60)

    tmp_dest = f"{dest}.tmp"
    with open(tmp_dest, "wb") as f:
        f.write(data)

    try:
        import rasterio

        with rasterio.open(tmp_dest) as ds:
            ds.read(1, window=((0, 1), (0, 1)))
    except Exception:
        os.remove(tmp_dest)
        raise

    os.replace(tmp_dest, dest)
    return dest


def count_cached_jrc_tiles() -> tuple[int, int] | None:
    """(cached_count, total_tiles) for the RP100 reclass set, or None if the tile
    index itself isn't cached yet (collect() hasn't completed even its first step).
    Used by FloodRiskHistoricalCollector.data_status() to report download progress
    without touching the network."""
    index_path = jrc_tile_extents_cache_path()
    if not os.path.exists(index_path):
        return None
    tiles = load_jrc_tile_index(index_path)
    cached = sum(1 for t in tiles if os.path.exists(jrc_tile_cache_path(t["id"], t["name"])))
    return cached, len(tiles)


def build_jrc_mosaic_grid(step_deg: float = JRC_MOSAIC_GRID_STEP_DEG):
    """Full-globe cell-center (lat, lon) axes for a flood_risk mosaic -- shared by
    BOTH modes (also used by FloodRiskLiveCollector for the MODIS mosaic; not
    renamed to something mode-neutral since JRC was here first and there's no
    behavioural difference to name around). Lat descends from north to south
    (matching both JRC's and MODIS's own per-tile GeoTIFF row order -- north-up,
    row 0 = top) so each tile's resampled data drops into the mosaic without a
    flip."""
    n_lat = round(180.0 / step_deg)
    n_lon = round(360.0 / step_deg)
    lat = 90.0 - step_deg / 2.0 - np.arange(n_lat) * step_deg
    lon = -180.0 + step_deg / 2.0 + np.arange(n_lon) * step_deg
    return lat, lon


def tile_dst_window(tile_bounds, step_deg: float = JRC_MOSAIC_GRID_STEP_DEG):
    """(row_start, row_end, col_start, col_end) into a build_jrc_mosaic_grid() array
    for the given tile's (lon_min, lat_min, lon_max, lat_max) bounds. Both JRC's and
    MODIS's tiling schemes are a fixed 10x10deg block per tile, so the window size is
    derived from the tiling scheme directly rather than re-measured per tile."""
    lon_min, _lat_min, _lon_max, lat_max = tile_bounds
    row_start = round((90.0 - lat_max) / step_deg)
    col_start = round((lon_min - (-180.0)) / step_deg)
    n = round(10.0 / step_deg)
    return row_start, row_start + n, col_start, col_start + n


def reproject_categorical_max(
    tile_path: str, dst_lat, dst_lon, remap, *, max_source_pixels: int | None = None
) -> np.ndarray:
    """Public (no leading underscore): also used by lib/vegetation_mask.py, whose
    Zenodo-mirrored MODIS Land Cover GeoTIFF needs the exact same "reproject a
    single-band categorical raster onto an arbitrary destination grid" mechanics,
    despite being an unrelated data source -- there's nothing flood/JRC-specific
    about the mechanics themselves, only about each caller's own `remap` rule.

    max_source_pixels: confirmed live (not a hypothetical) that omitting this reads
    the WHOLE band 1 into memory before remap/reproject even run -- fine for JRC/
    MODIS-flood's ~10x10deg tiles (small enough to read wholesale), but the
    vegetation mask's source is one global ~86400x35849 mosaic (~3.1 billion
    pixels uncompressed); reading that wholesale OOM-killed the process at ~4GB RSS
    for even a modest 180x360 destination grid. When set and the source has more
    pixels than this, the source is read via a decimated `out_shape` (nearest,
    cheap) chosen so its resolution comfortably exceeds the destination's, rather
    than reading every native pixel just to immediately discard nearly all of them
    in the max-resample. This is a real accuracy/memory trade-off: a small
    burnable patch entirely within pixels skipped by decimation could be missed --
    acceptable here because the destination grids that pass this are themselves far
    coarser than the decimated read, and the whole mask is inherently a coarse
    sanity filter, not a precision boundary. Existing callers omit this and keep
    doing a full wholesale read (their tiles are always small enough for it to be
    correct and cheap) -- passing it is opt-in, so their behaviour is unchanged.

    Shared rasterio.warp.reproject(Resampling.max) mechanics behind
    resample_jrc_tile_onto_grid and resample_modis_flood_tile_onto_grid: read band 1
    of a single-band categorical GeoTIFF, apply `remap(source_array, src_dataset)`
    (each caller's own nodata-zeroing / reclassification rule) to the SOURCE array
    BEFORE reprojecting, then max-resample onto the given destination cell-center
    axes -- typically the exact sub-window covering one tile's 10x10deg footprint
    (see tile_dst_window). Resampling.max, not average/nearest: categorical hazard/
    detection data must never let a coarse working-resolution cell hide a known
    worst-case within it (same reasoning as coastline.py's _rasterize_land_mask
    uses exact rasterization for categorical land/sea data).

    Assumes dst_lat is north-first (descending) -- every existing caller's mosaic
    grid is built that way (see build_jrc_mosaic_grid). A caller with an ascending
    axis must flip it before calling this and flip the result back afterward (see
    lib/vegetation_mask.py's burnable_vegetation_mask for why that's a real case,
    not a hypothetical one).

    Remapping happens in the SOURCE array rather than via GDAL's src_nodata/
    dst_nodata: confirmed live that GDAL's max/min/average-family resamplers only
    apply nodata masking to a destination cell that has SOME valid contributing
    source pixels -- a destination cell whose contributing source window is
    entirely nodata reprojects the raw nodata value through unmasked instead of
    yielding dst_nodata, silently turning "unclassified"/"insufficient data" into a
    spurious "worse-than-any-real-category" reading under max-resampling. Handling
    it in the source array sidesteps the bug entirely.
    """
    import rasterio
    from rasterio import Affine
    from rasterio.warp import Resampling, reproject

    step_lat = float(dst_lat[0] - dst_lat[1]) if len(dst_lat) > 1 else JRC_MOSAIC_GRID_STEP_DEG
    step_lon = float(dst_lon[1] - dst_lon[0]) if len(dst_lon) > 1 else JRC_MOSAIC_GRID_STEP_DEG
    dst_transform = Affine(
        step_lon, 0.0, float(dst_lon[0]) - step_lon / 2.0,
        0.0, -step_lat, float(dst_lat[0]) + step_lat / 2.0,
    )
    dst = np.zeros((len(dst_lat), len(dst_lon)), dtype=np.uint8)

    with rasterio.open(tile_path) as src:
        src_transform = src.transform
        src_pixels = src.width * src.height
        if max_source_pixels and src_pixels > max_source_pixels:
            scale = (src_pixels / max_source_pixels) ** 0.5
            out_width = max(1, int(src.width / scale))
            out_height = max(1, int(src.height / scale))
            band = src.read(
                1, out_shape=(out_height, out_width), resampling=Resampling.nearest
            )
            src_transform = src.transform * src.transform.scale(
                src.width / out_width, src.height / out_height
            )
        else:
            band = src.read(1)
        source = remap(band, src)
        reproject(
            source=source,
            destination=dst,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.max,
        )
    return dst


def resample_jrc_tile_onto_grid(tile_path: str, dst_lat, dst_lon) -> np.ndarray:
    """Downsample one JRC reclass tile (uint8, ~90m native resolution, categories
    1-4 + nodata=255) onto the given destination cell-center axes. Native nodata
    (255 -- areas JRC's model didn't classify, not necessarily hazard-free) is
    collapsed to 0 (this mosaic's own default fill for land outside any tile's
    footprint) before reprojecting -- see reproject_categorical_max's docstring
    for why."""

    def _zero_nodata(source, src):
        if src.nodata is not None:
            source = source.copy()
            source[source == src.nodata] = 0
        return source

    return reproject_categorical_max(tile_path, dst_lat, dst_lon, _zero_nodata)


def save_jrc_hazard_mosaic(path: str, band: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    """Atomic write of a final assembled global categorical mosaic -- shared by
    both modes (also used for the MODIS flood mosaic; the format has no JRC-specific
    content, just a (band, latitude, longitude) netCDF)."""
    import xarray as xr

    ds = xr.Dataset(
        {"band": (("latitude", "longitude"), band.astype(np.uint8))},
        coords={"latitude": lat.astype(np.float64), "longitude": lon.astype(np.float64)},
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    ds.to_netcdf(tmp_path)
    os.replace(tmp_path, path)


def load_jrc_hazard_mosaic(path: str):
    """(band, lat, lon) arrays from a cached hazard mosaic -- `band` is uint8.
    Shared by both modes; see save_jrc_hazard_mosaic."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        band = ds["band"].values.astype(np.uint8)
        lat = ds["latitude"].values.astype(np.float64)
        lon = ds["longitude"].values.astype(np.float64)
    return band, lat, lon


# --- NASA LANCE MODIS Flood Product (Live mode: "Observed Current Inundation") ---
#
# Replaces the original GloFAS ensemble-discharge-FORECAST design entirely (see
# collectors/flood_risk.py's module docstring for the full history): repeated real
# OOM crashes and severe, unfixable network flakiness against ECMWF/Copernicus's
# shared object-store backend made GloFAS Live unmaintainable on this host, and a
# forecast product was never really comparable to Historical's hazard-potential map
# anyway. MODIS's flood product instead reports OBSERVED current inundation --
# actual satellite-detected flooding, refreshed continuously, not a forward-looking
# risk estimate -- a genuinely different (and simpler-to-integrate) kind of data.
#
# MCDWD_L3_F1C_NRT: the 1-Day cloud-shadow-screened flood layer, MODIS Terra+Aqua,
# Release 1 (not VIIRS's own VCDWD/VCDWDG product, which is still Beta 1 and has no
# GeoTIFF output yet -- HDF5 only), delivered as a single-band GeoTIFF per 10x10deg
# tile (287 land-relevant tiles globally, not full ocean coverage). See the official
# "LANCE MODIS/VIIRS Flood Product User Guide" (Rev E, 22 Apr 2025) -- pixel values
# (that guide's Table 7): 0 no water, 1 surface water (matches reference), 2
# recurring flood (not yet populated by NASA as of this writing), 3 flood (unusual,
# outside the reference-water mask), 255 insufficient data. Only value 3 is
# rendered (see resample_modis_flood_tile_onto_grid) -- normal water bodies are
# already visible on the basemap, and 255 must never be conflated with a confirmed
# flood detection.
MODIS_FLOOD_SHORTNAME = "MCDWD_L3_F1C_NRT"
MODIS_FLOOD_COLLECTION = "61"
MODIS_FLOOD_VALUE = 3  # the "Flood (unusual)" pixel code -- see Table 7 above
MODIS_FLOOD_TILE_STEP_DEG = 10.0

LANCE_BASE_URL = "https://nrt3.modaps.eosdis.nasa.gov"
_LANCE_LISTING_URL = f"{LANCE_BASE_URL}/api/v2/content/details"
_LANCE_ARCHIVE_URL = f"{LANCE_BASE_URL}/archive/allData/{MODIS_FLOOD_COLLECTION}/{MODIS_FLOOD_SHORTNAME}"

# A cached tile whose last successful download is older than this is dropped from
# the mosaic on the next rebuild rather than kept showing as "current" flooding --
# an "Observed CURRENT Inundation" claim must not keep rendering days-old flood
# pixels through a LANCE outage. Double the 1-Day composite's own ~24h freshness
# window, for margin against a single missed cycle.
MODIS_FLOOD_STALE_S = 48 * 3600

# <SHORTNAME>.A<DATE>.h<HH>v<VV>.<COLLECTION>.tif -- confirmed live against a
# real LANCE listing response: unlike the production-timestamp-suffixed HDF
# example the User Guide (section 6.2) shows, the real GeoTIFF filenames this
# API actually returns carry NO trailing production-timestamp segment (e.g.
# "MCDWD_L3_F1C_NRT.A2026247.h09v01.061.tif") -- a stricter regex requiring
# one here silently matched zero of 287 real entries.
_MODIS_FILENAME_RE = re.compile(
    r"^[A-Za-z0-9_]+\.A(?P<date>\d{7})\.h(?P<h>\d{2})v(?P<v>\d{2})\.\d{3}\.tif$"
)


def modis_flood_tile_bounds(h: int, v: int) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) for MODIS/Land's standard 10x10deg
    Linear Lat/Lon tiling scheme (h counts east from -180, v counts south from +90
    -- https://modis-land.gsfc.nasa.gov/MODLAND_grid.html), reused as-is by the
    flood product (User Guide Table 5)."""
    lon_min = -180.0 + h * MODIS_FLOOD_TILE_STEP_DEG
    lat_max = 90.0 - v * MODIS_FLOOD_TILE_STEP_DEG
    return lon_min, lat_max - MODIS_FLOOD_TILE_STEP_DEG, lon_min + MODIS_FLOOD_TILE_STEP_DEG, lat_max


def _modis_flood_cache_dir() -> str:
    return os.path.join(_flood_risk_cache_dir(), "modis_flood")


def modis_flood_tile_cache_path(h: int, v: int) -> str:
    return os.path.join(_modis_flood_cache_dir(), f"h{h:02d}v{v:02d}.tif")


def _modis_flood_tile_meta_path(h: int, v: int) -> str:
    """Sidecar recording the FILENAME (not just bytes) of the tile currently cached
    at modis_flood_tile_cache_path(h, v) -- the cache path itself is fixed per
    tile, but MODIS's own filename embeds the tile's data date (<A2026247>-style),
    so comparing this sidecar's content against a fresh LANCE listing's filename
    is how modis_flood_tile_is_current() detects "this tile's
    content actually changed" without re-downloading it first."""
    return modis_flood_tile_cache_path(h, v) + ".name"


def modis_flood_mosaic_cache_path(workdir: str) -> str:
    """Cache path for the mosaic FloodRiskLiveCollector rebuilds every cycle it
    finds a changed or newly-expired tile -- under {workdir}/data (bind-mounted),
    same convention as jrc_hazard_mosaic_cache_path. The render task (tasks/
    flood_risk.py) treats a fresh mtime here as its own "new data" signal, exactly
    like Historical mode's mosaic file."""
    return os.path.join(workdir, "data", "flood_risk_cache_modis_flood_mosaic.nc")


def modis_flood_listing_url(date) -> str:
    """LANCE's JSON file-listing API for one UTC calendar day (User Guide section
    6.1.1) -- a single small JSON response covering every currently-produced
    tile's filename, used both to discover which tiles exist right now and, via
    modis_flood_tile_is_current(), to tell which ones actually changed since
    last cycle. `date` is a datetime
    (only .year and .timetuple().tm_yday are used)."""
    day_of_year = date.timetuple().tm_yday
    temporal_range = f"{date.year}-{day_of_year:03d}"
    return (
        f"{_LANCE_LISTING_URL}?products={MODIS_FLOOD_SHORTNAME}"
        f"&archiveSets={MODIS_FLOOD_COLLECTION}&temporalRanges={temporal_range}"
    )


def parse_modis_flood_listing(payload: dict) -> list[dict]:
    """Normalize LANCE's JSON listing response into
    [{"h", "v", "filename", "download_url"}, ...], one entry per tile file for the
    requested day -- entries whose filename doesn't match the documented MODIS
    flood-product grammar (_MODIS_FILENAME_RE) are skipped rather than raising, so
    an unrelated/unexpected entry can't abort the whole listing.

    Confirmed live against a real Earthdata-token-authenticated response: the
    top-level payload is a dict wrapping the file-detail array under "content"
    (NOT a bare JSON array, despite how User Guide section 6.1.1 reads) --
    payload["content"][i]["name"] and ["downloadsLink"] are exactly the real
    field names, so both fallbacks below are now confirmed-unused belt-and-
    braces rather than guesses."""
    tiles = []
    for entry in payload.get("content", []):
        name = entry.get("name") or entry.get("fileName") or ""
        m = _MODIS_FILENAME_RE.match(name)
        if not m:
            continue
        download_url = entry.get("downloadsLink") or entry.get("downloadLink")
        if not download_url:
            date_str = m.group("date")
            download_url = f"{_LANCE_ARCHIVE_URL}/{date_str[:4]}/{date_str[4:]}/{name}"
        tiles.append(
            {
                "h": int(m.group("h")),
                "v": int(m.group("v")),
                "filename": name,
                "download_url": download_url,
            }
        )
    return tiles


def resolve_earthdata_token(label: str) -> str | None:
    """The NASA Earthdata Login bearer token used for LANCE downloads, or None
    (having logged why) if EARTHDATA_TOKEN isn't configured. No config-driven base
    URL to also check: LANCE_BASE_URL is hardcoded, same "one real endpoint, not
    user-swappable" convention as JRC_BASE_URL."""
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token:
        logger.warning(f"{label}: no EARTHDATA_TOKEN configured; skipping.")
        return None
    return token


def fetch_modis_flood_listing(date, token: str) -> list[dict]:
    """GET + parse LANCE's JSON tile listing for one UTC calendar day. Raises on a
    network/HTTP error -- callers treat "listing unavailable" as "nothing new to
    render this cycle", same graceful-fallback contract as the JRC helpers above."""
    import requests

    r = requests.get(
        modis_flood_listing_url(date),
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    return parse_modis_flood_listing(r.json())


def modis_flood_tile_is_current(tile: dict) -> bool:
    """Whether the cached copy of `tile` (an entry from parse_modis_flood_listing)
    already matches the listing's filename -- i.e. this tile's content has NOT
    changed since it was last downloaded, so re-fetching it would be wasted work."""
    try:
        with open(_modis_flood_tile_meta_path(tile["h"], tile["v"])) as f:
            return f.read().strip() == tile["filename"]
    except FileNotFoundError:
        return False


def ensure_modis_flood_tile_cached(tile: dict, token: str) -> str:
    """Download one MODIS flood GeoTIFF tile (single 8-bit band, typically tens of
    KB compressed), overwriting any existing cache file for this (h, v)
    unconditionally -- unlike JRC's tiles (a fetch-once-forever static hazard map),
    a MODIS tile's content genuinely changes day to day, so the caller
    (FloodRiskLiveCollector) only calls this once modis_flood_tile_is_current() has
    already said this tile's remote filename changed. Returns the cached .tif
    path. Raises on failure -- caller skips this tile for the current cycle and
    keeps whatever was cached before, same graceful-fallback contract as
    ensure_jrc_tile_cached."""
    from atmos_gl.lib.gfs import download_whole

    dest = modis_flood_tile_cache_path(tile["h"], tile["v"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = download_whole(
        tile["download_url"], headers={"Authorization": f"Bearer {token}"}
    )

    tmp_dest = f"{dest}.tmp"
    with open(tmp_dest, "wb") as f:
        f.write(data)

    try:
        import rasterio

        with rasterio.open(tmp_dest) as ds:
            ds.read(1, window=((0, 1), (0, 1)))
    except Exception:
        os.remove(tmp_dest)
        raise

    os.replace(tmp_dest, dest)
    with open(_modis_flood_tile_meta_path(tile["h"], tile["v"]), "w") as f:
        f.write(tile["filename"])
    return dest


def prune_stale_modis_flood_tiles() -> bool:
    """Delete any cached MODIS flood tile (+ its .name sidecar) whose last
    successful download is older than MODIS_FLOOD_STALE_S -- see that constant's
    docstring. Returns True if anything was pruned: the caller
    (FloodRiskLiveCollector.has_new_data) treats a prune as "the mosaic needs a
    rebuild even though nothing NEW arrived", since dropping a tile changes the
    mosaic just as much as adding one."""
    cache_dir = _modis_flood_cache_dir()
    if not os.path.isdir(cache_dir):
        return False
    now = time.time()
    pruned = False
    for name in os.listdir(cache_dir):
        if not name.endswith(".tif"):
            continue
        path = os.path.join(cache_dir, name)
        if now - os.path.getmtime(path) > MODIS_FLOOD_STALE_S:
            os.remove(path)
            meta_path = path + ".name"
            if os.path.exists(meta_path):
                os.remove(meta_path)
            pruned = True
    return pruned


def cached_modis_flood_tiles() -> list[tuple[int, int, str]]:
    """[(h, v, tile_path), ...] for every MODIS flood tile currently on disk (after
    pruning), parsed from the cache filenames themselves -- used to rebuild the
    mosaic from whatever's cached, independent of which tiles happened to be
    re-downloaded THIS cycle (a tile that failed to refresh this cycle still
    contributes its last-known-good content, same resilience JRC's per-tile cache
    already provides)."""
    cache_dir = _modis_flood_cache_dir()
    if not os.path.isdir(cache_dir):
        return []
    tiles = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".tif"):
            continue
        m = re.match(r"^h(\d{2})v(\d{2})\.tif$", name)
        if not m:
            continue
        tiles.append((int(m.group(1)), int(m.group(2)), os.path.join(cache_dir, name)))
    return tiles


# A "Flood (unusual)" pixel more than this many source pixels (~231m each at
# MODIS's native resolution) from the nearest "Surface water" (value 1) pixel is
# dropped rather than rendered -- confirmed live against cached NZ tiles
# (h34v13/h35v12/h35v13) that isolated flood pixels cluster in steep terrain
# (Fiordland, Aoraki/Mt Cook, the Rotorua/Taupo hill country, the Wellington
# hills) with no nearby mapped water at all: this product is "cloud-shadow-
# screened" but NOT terrain-shadow-screened, so low-sun-angle terrain shadow in
# mountainous regions is misclassified as flood. Requiring proximity to known
# water is a heuristic, not a precise fix (a real flood does start from/spread
# around existing water, but so does a shadow pooling in the same valley) --
# see the "Threshold on NASA Live flood" decision this radius came from for the
# fuller trade-off discussion. 3px (~700m) trades keeping small real
# river-adjacent floods against still admitting some shadow-near-a-stream noise.
MODIS_FLOOD_WATER_ADJACENCY_PX = 3


def resample_modis_flood_tile_onto_grid(tile_path: str, dst_lat, dst_lon) -> np.ndarray:
    """Downsample one MODIS flood GeoTIFF tile onto the given destination
    cell-center axes, binarizing to the "binary flood overlay" visualization
    decision on the way in: pixel value MODIS_FLOOD_VALUE (3, "Flood (unusual)")
    becomes 1 only if within MODIS_FLOOD_WATER_ADJACENCY_PX of a "Surface water"
    (value 1) pixel -- see that constant's docstring for why. Everything else (0
    no-water, 1 surface water itself, 2 recurring-flood [not yet populated], 255
    insufficient data) becomes 0 -- see reproject_categorical_max's docstring
    for why remapping happens in the source array."""

    def _binarize_flood(source, _src):
        from scipy.ndimage import binary_dilation

        flood = source == MODIS_FLOOD_VALUE
        water = source == 1
        nearby_water = binary_dilation(water, iterations=MODIS_FLOOD_WATER_ADJACENCY_PX)
        return (flood & nearby_water).astype(np.uint8)

    return reproject_categorical_max(tile_path, dst_lat, dst_lon, _binarize_flood)
