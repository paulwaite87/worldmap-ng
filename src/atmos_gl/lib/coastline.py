#!/usr/bin/env python3
"""Coastline land-masking, split out of tasks/common.py (architecture review
candidate "tasks/common.py bundles six unrelated concerns"): a pure geometry
function with no Updater/MapRegion coupling. `coastline_land_mask` is called
directly by sst.py/greenhouse_gases.py; currents.py and waves.py instead go through
LandMaskCache + nearest_fill_and_regrid_uv below, which used to be byte-identical
code duplicated in both files (waves.py's own comment said so: "Mirrors currents.py's
_land_mask_for exactly") -- see architecture review candidate "share currents' and
waves' land-mask/regrid pipeline".
"""
import logging
import os
import zipfile
from io import BytesIO

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

logger = logging.getLogger(__name__)

# GSHHG (Global Self-consistent, Hierarchical, High-resolution Geography Database)
# replaces the previous Natural Earth-based coastline masking -- see docs/adr/0011
# (superseded) and docs/adr/0013. Natural Earth's "10m" designation is a *cartographic
# scale* (1:10,000,000), not survey precision, and produced a genuinely noisy "swiss
# cheese" land mask on complex, convoluted coastlines (Northland NZ, Tasmania) even
# though it read fine on large, simple landmasses -- see ADR-0011's investigation.
# GSHHG's 'h' (high) tier is a real ~1:1,000,000-scale coastline dataset, fixing the
# misclassification.
#
# Distributed as a single ZIP bundling all 5 resolution tiers. The University of
# Hawaii mirror (soest.hawaii.edu, GSHHG's canonical distribution point) throttled to
# single-digit KB/s when measured from this deployment's network path (~4hr for the
# full 142MB); the Generic Mapping Tools project's GitHub Releases mirror (identical
# file, same version) serves it from GitHub's own CDN instead -- ~6MB/s, a ~25s
# one-time download.
_GSHHG_VERSION = "2.3.7"
_GSHHG_TIER = "h"
_GSHHG_URL = (
    f"https://github.com/GenericMappingTools/gshhg-gmt/releases/download/"
    f"{_GSHHG_VERSION}/gshhg-shp-{_GSHHG_VERSION}.zip"
)

# In-memory cache of the (bbox-clipped) land geometry per rounded bbox, so a render
# reuses the same clipped geometry across hours and across layers within one worker
# process's lifetime. Keyed only by bbox now (no more per-resolution tiers -- GSHHG
# 'h' is the one dataset every caller shares). Separate from, and layered on top of,
# _load_gshhg_land_union()'s own on-disk cache of the raw global union below.
_COAST_GEOM_CACHE = {}


def _gshhg_cache_dir() -> str:
    """Container-local cache dir -- NOT bind-mounted, mirroring how cartopy's own
    Natural Earth downloads were already cached (~/.local/share/cartopy): ephemeral
    across container recreation, but persistent across worker-process restarts within
    one container's lifetime (worker processes share the container filesystem)."""
    return os.path.join(os.path.expanduser("~"), ".local", "share", "gshhg")


def _gshhg_shapefile_path() -> str:
    return os.path.join(
        _gshhg_cache_dir(), "GSHHS_shp", _GSHHG_TIER, f"GSHHS_{_GSHHG_TIER}_L1.shp"
    )


def _download_gshhg_if_needed() -> str:
    """Download + extract just the 'h' tier land layer (L1: coastline, GSHHG's own
    layer-numbering convention) from the GSHHG bundle, if not already cached on disk.
    Returns the .shp path. Raises on failure -- callers catch and fall back gracefully,
    same contract the old Natural Earth path had."""
    shp_path = _gshhg_shapefile_path()
    if os.path.exists(shp_path):
        return shp_path

    from atmos_gl.lib.gfs import download_whole

    cache_dir = _gshhg_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Downloading GSHHG '{_GSHHG_TIER}' coastline data (one-time)...")
    data = download_whole(_GSHHG_URL, timeout=180)
    prefix = f"GSHHS_shp/{_GSHHG_TIER}/GSHHS_{_GSHHG_TIER}_L1."
    with zipfile.ZipFile(BytesIO(data)) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix)]
        zf.extractall(cache_dir, members)
    return shp_path


def _gshhg_land_union_cache_path() -> str:
    return os.path.join(_gshhg_cache_dir(), f"gshhg_{_GSHHG_TIER}_land_union.wkb")


def _load_gshhg_land_union():
    """The unioned GSHHG land geometry (global), loaded from a disk cache if present.

    The raw load+repair+union pipeline -- 144,749 polygons at 'h' resolution, some of
    which are invalid by shapely's strict standards and need shapely.make_valid()
    before unary_union can even succeed -- measured ~227s in this deployment, over 10x
    Natural Earth's own ~21s equivalent (see docs/adr/0011). Under a per-worker-process
    in-memory-only cache, every fresh or crash-recycled render worker would pay that
    cost before rendering any layer needing a land mask. So the FINAL unioned result is
    additionally cached to disk (WKB) after the first successful build: only the very
    first worker ever to need it pays the ~227s cost; every worker after that just
    deserializes the cached geometry, a cheap operation.

    Raises on failure (network/parse/topology) -- callers catch and log, same
    graceful-fallback contract as coastline_land_mask()'s own try/except.
    """
    import shapely

    union_cache_path = _gshhg_land_union_cache_path()
    if os.path.exists(union_cache_path):
        with open(union_cache_path, "rb") as f:
            return shapely.from_wkb(f.read())

    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union

    shp_path = _download_gshhg_if_needed()
    reader = shpreader.Reader(shp_path)
    # GSHHG's raw polygons include some that are invalid by shapely's strict standards
    # (self-intersections/degenerate rings) -- unary_union raises a TopologyException
    # outright without this repair pass. Natural Earth's much smaller, cleaner feature
    # set never needed it.
    geoms = [shapely.make_valid(g) for g in reader.geometries()]
    land = unary_union(geoms)

    with open(union_cache_path, "wb") as f:
        f.write(shapely.to_wkb(land))
    return land


def _rasterize_land_mask(land_geom, mesh_lon, mesh_lat):
    """Rasterize land_geom (a GSHHG land union Polygon/MultiPolygon, already
    bbox-clipped by the caller) directly onto the given regular lon/lat mesh, via
    GDAL's polygon rasterizer (rasterio.features.rasterize) rather than per-point
    shapely.contains_xy testing.

    Per-point testing doesn't scale to a fine global grid: walking every one of a
    full-globe 0.02deg grid's ~162M points individually against the ~145K-feature
    GSHHG union ran 8+ minutes and 3.4GB+ RSS without finishing, crashing the render
    worker (BrokenProcessPool) -- confirmed live when _WAVES_REGRID_STEP_DEG was
    dropped from 0.08 to 0.02. GDAL's rasterizer instead burns the polygon boundary
    onto the grid directly (cost bound by polygon edge count + grid size, the way a
    real rasterizer scales), not by grid-point count times polygon complexity.

    rasterize()'s default all_touched=False burns a pixel when its CENTRE falls
    inside the geometry -- the transform below is built so pixel (0, 0)'s centre
    lands exactly on (lon_1d[0], lat_1d[0]), reproducing the same per-point sample
    contains_xy used to test, just computed by an algorithm that scales.
    """
    from rasterio import Affine
    from rasterio.features import rasterize

    if getattr(land_geom, "is_empty", False):
        return np.zeros(mesh_lon.shape, dtype=bool)

    lon_1d = mesh_lon[0, :]
    lat_1d = mesh_lat[:, 0]
    width = lon_1d.size
    height = lat_1d.size

    dlon = float(lon_1d[1] - lon_1d[0]) if width > 1 else 1.0
    dlat = float(lat_1d[1] - lat_1d[0]) if height > 1 else 1.0
    lon0 = float(lon_1d[0])
    lat0 = float(lat_1d[0])
    transform = Affine(dlon, 0.0, lon0 - dlon / 2.0, 0.0, dlat, lat0 - dlat / 2.0)

    mask = rasterize(
        [(land_geom, 1)],
        out_shape=(int(height), int(width)),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def coastline_land_mask(
    mesh_lon, mesh_lat, lon_min, lat_min, lon_max, lat_max, dilate: bool = True
):
    """Boolean land mask (True over land) sampled at the given mesh, cut from true
    GSHHG 'h' coastline geometry (see docs/adr/0013).

    Shared by any layer that needs to remove land from an ocean field (currents, sst,
    waves, greenhouse_gases): a data-derived NaN mask only knows where the model lacked
    data, so model values can smear up to the interpolation cap onto the coast; cutting
    against real coastline polygons clips the field to the actual shoreline. Returns
    None if the geometry can't be loaded (e.g. no network for the one-time GSHHG
    download) so callers can fall back to whatever data-derived mask they have.

    dilate=True (the default) grows the land boundary by one cell (8-connectivity, so
    diagonal-only coastal corners are covered too) before returning. Every current
    caller feeds this mask into a LINEAR-filtered GPU texture with a hard alpha>=0.5
    discard (createFillLayer/createStaticFillLayer, ui/modules/_webglfill.js), which
    blends colour/velocity across every texel boundary -- including the true coastline
    edge -- so a sample landing more than half-way toward the (alpha=255) ocean side
    survives the discard with bleed-through colour even on the land side of the real
    coastline (confirmed live: 99.9% mask/pixel agreement pre-fix, with the mismatch
    concentrated exactly at coastlines; see docs/adr/0014). Growing the mask by ~1 texel
    pushes the boundary far enough offshore that this blend zone never crosses back onto
    land, at the cost of one texel's width of true coastal water going uncoloured too.
    Pass dilate=False for a caller that wants the raw geometric classification instead
    (e.g. a non-GPU consumer, or a test isolating rasterization from this post-process).

    This used to be a per-caller step (sst.py, currents.py, waves.py each ran their own
    identical binary_dilation, and greenhouse_gases.py's own caller was missed entirely
    -- see docs/adr/0014's "Revisit if" clause, which only anticipated currents/waves)
    -- now every caller gets it correctly, by construction, for free.
    """
    try:
        key = (
            round(lon_min, 2),
            round(lat_min, 2),
            round(lon_max, 2),
            round(lat_max, 2),
        )
        land_geom = _COAST_GEOM_CACHE.get(key)
        if land_geom is None:
            land_geom = _load_gshhg_land_union()
            # Every current caller passes the global bbox (-180/-90/180/90), for which
            # clipping is a no-op -- skip it there rather than pay for an intersection
            # against a huge unioned geometry for zero benefit. Kept correct for a
            # hypothetical narrower-bbox caller.
            if (lon_min, lat_min, lon_max, lat_max) != (-180.0, -90.0, 180.0, 90.0):
                from shapely.geometry import box

                land_geom = land_geom.intersection(box(lon_min, lat_min, lon_max, lat_max))
            _COAST_GEOM_CACHE[key] = land_geom

        mask = _rasterize_land_mask(land_geom, mesh_lon, mesh_lat)
        if dilate:
            mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
        return mask
    except Exception as exc:  # network/data/parse failure -> graceful fallback
        logger.warning(
            f"Coastline geometry unavailable ({exc!r}); land mask skipped."
        )
        return None


class LandMaskCache:
    """Global (-180/-90/180/90) coastline land mask, cached per regridded-grid shape
    for the life of one run -- shared by CurrentsUpdater and WavesUpdater, whose own
    per-class caching wrapper around coastline_land_mask() used to be byte-identical.
    `label` distinguishes each caller's log lines ("Currents"/"Waves"); a shape that
    fails once (geometry unavailable) is cached as None too, so it doesn't retry every
    call within the same run."""

    def __init__(self, label: str):
        self._label = label
        self._cache = {}

    def get(self, lat, lon, shape):
        if shape in self._cache:
            return self._cache[shape]
        mesh_lon, mesh_lat = np.meshgrid(np.asarray(lon), np.asarray(lat))
        land = coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)
        self._cache[shape] = land
        if land is not None:
            logger.info(
                f"{self._label}: built {shape} coastline land mask "
                f"({int(land.sum())} land cells cut)."
            )
        return land


def nearest_fill_and_regrid_uv(regrid_fn, u_native, v_native, lat_native, lon_native, step_deg):
    """Nearest-fill native NaN (land/no-data cells) in u/v, then regrid both to
    `step_deg` via `regrid_fn` (an Updater.regrid_for_lod-shaped callable) -- shared by
    CurrentsUpdater.plot() and WavesUpdater._masked_uv(), whose own copies of this
    sequence used to be byte-identical apart from the regrid step constant.

    The nearest-fill happens BEFORE regridding so bilinear interpolation doesn't bleed
    NaN outward from the coast into legitimate near-shore water -- the true coastline
    cut (LandMaskCache, applied by the caller afterward) is what actually determines
    land/sea, not this fill. A field with no valid cells at all is left untouched
    (distance_transform_edt needs SOME valid data to fill from).

    Does not mutate the caller's u_native/v_native arrays. Returns
    (new_lats, new_lons, u, v) -- callers apply any of their own steps (e.g. currents'
    speed-minimum threshold) and the land-mask cut themselves, since ordering differs
    between callers.
    """
    u_native = np.asarray(u_native, dtype=np.float32).copy()
    v_native = np.asarray(v_native, dtype=np.float32).copy()

    for native in (u_native, v_native):
        bad = ~np.isfinite(native)
        if bad.any() and not bad.all():
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            native[:] = native[tuple(idx)]

    new_lats, new_lons, u = regrid_fn(
        u_native, lat_native, lon_native, fill_value=np.nan, step_override=step_deg
    )
    _, _, v = regrid_fn(
        v_native, lat_native, lon_native, fill_value=np.nan, step_override=step_deg
    )
    return new_lats, new_lons, u, v
