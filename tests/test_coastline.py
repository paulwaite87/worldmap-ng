#!/usr/bin/env python3
"""Tests for lib/coastline.py's LandMaskCache and nearest_fill_and_regrid_uv --
architecture review candidate "share currents' and waves' land-mask/regrid pipeline".
CurrentsUpdater._land_mask_for and WavesUpdater._land_mask_for were byte-identical
(waves.py's own docstring said so: "Mirrors currents.py's _land_mask_for exactly"),
as was the nearest-fill-then-regrid block ahead of each class's land-mask cut. These
tests lock the shared behavior directly; each caller's own test file only needs to
confirm it's wired up, not re-test the logic.
"""
from unittest.mock import MagicMock, mock_open, patch, call

import pytest
import numpy as np
from shapely.geometry import Point, box

from atmos_gl.lib import coastline as coastline_mod
from atmos_gl.lib.coastline import (
    LandMaskCache,
    coastline_land_mask,
    nearest_fill_and_regrid_uv,
)


@pytest.fixture(autouse=True)
def _clear_coast_geom_cache():
    """_COAST_GEOM_CACHE is module-level state, keyed by bbox -- every real caller
    shares the same global bbox key, so an earlier test's cached entry would otherwise
    leak into later tests that mock _load_gshhg_land_union differently."""
    coastline_mod._COAST_GEOM_CACHE.clear()
    yield
    coastline_mod._COAST_GEOM_CACHE.clear()


# ---------------------------------------------------------------------------
# LandMaskCache
# ---------------------------------------------------------------------------

def test_get_caches_per_shape():
    cache = LandMaskCache("Test")
    sentinel = np.array([[True, False]])
    with patch(
        "atmos_gl.lib.coastline.coastline_land_mask", return_value=sentinel
    ) as mock_coast:
        first = cache.get(lat=[0.0], lon=[0.0, 1.0], shape=(1, 2))
        second = cache.get(lat=[0.0], lon=[0.0, 1.0], shape=(1, 2))

    assert first is sentinel
    assert second is sentinel
    mock_coast.assert_called_once()


def test_get_uses_a_separate_cache_entry_per_distinct_shape():
    cache = LandMaskCache("Test")
    with patch(
        "atmos_gl.lib.coastline.coastline_land_mask",
        side_effect=[np.array([[True]]), np.array([[False, False]])],
    ) as mock_coast:
        cache.get(lat=[0.0], lon=[0.0], shape=(1, 1))
        cache.get(lat=[0.0], lon=[0.0, 1.0], shape=(1, 2))

    assert mock_coast.call_count == 2


def test_get_passes_the_global_bbox():
    cache = LandMaskCache("Test")
    with patch(
        "atmos_gl.lib.coastline.coastline_land_mask", return_value=None
    ) as mock_coast:
        cache.get(lat=[0.0], lon=[0.0], shape=(1, 1))

    args = mock_coast.call_args.args
    assert args[2:] == (-180.0, -90.0, 180.0, 90.0)


def test_get_caches_none_too_when_geometry_is_unavailable():
    """Matches the pre-extraction behavior exactly: a None result (geometry load
    failure) is cached like any other value, so a shape that failed once doesn't
    retry every subsequent call within the same run."""
    cache = LandMaskCache("Test")
    with patch(
        "atmos_gl.lib.coastline.coastline_land_mask", return_value=None
    ) as mock_coast:
        first = cache.get(lat=[0.0], lon=[0.0], shape=(1, 1))
        second = cache.get(lat=[0.0], lon=[0.0], shape=(1, 1))

    assert first is None
    assert second is None
    mock_coast.assert_called_once()


# ---------------------------------------------------------------------------
# coastline_land_mask (GSHHG -- docs/adr/0013, supersedes docs/adr/0011's Natural
# Earth precision limitation)
# ---------------------------------------------------------------------------

def _square_land(lon_min, lat_min, lon_max, lat_max):
    """A small synthetic 'land' polygon -- stands in for the real (144,749-feature)
    GSHHG geometry so these tests stay fast and hermetic, exercising the real shapely
    contains_xy/intersection wiring without touching the network or a real shapefile."""
    return box(lon_min, lat_min, lon_max, lat_max)


def test_coastline_land_mask_classifies_points_against_the_gshhg_union():
    land = _square_land(0.0, 0.0, 10.0, 10.0)
    mesh_lon, mesh_lat = np.meshgrid([-5.0, 5.0], [-5.0, 5.0])
    # dilate=False: isolate the raw rasterize/classify step from the dilation
    # post-process (covered separately below) -- on this 2x2 mesh the default dilation
    # would spread the single land corner into its neighbours too.
    with patch("atmos_gl.lib.coastline._load_gshhg_land_union", return_value=land):
        mask = coastline_land_mask(
            mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0, dilate=False
        )

    # (5,5) is inside the square (land); the other three corners are outside (water).
    assert mask.tolist() == [[False, False], [False, True]]


# ---- coastal-bleed fix: dilate=True (the default) grows the land mask -------------
# See docs/adr/0014-dilate-sst-land-mask-for-linear-filtering-bleed.md. Used to be a
# per-caller step (sst.py/currents.py/waves.py each ran their own identical
# binary_dilation, and greenhouse_gases.py's caller was missed entirely) -- now lives
# once, here, so every caller (present and future) gets it correctly by construction.

def test_coastline_land_mask_dilates_by_default():
    # Land square (5..15) contains only the value-10 mesh point on each axis, which
    # lands at index [1,1] in this 4x4 mesh; dilation (8-connectivity) should grow
    # that single True cell into its full immediate neighbourhood, but not reach the
    # far corner two steps away.
    land = _square_land(5.0, 5.0, 15.0, 15.0)
    mesh_lon, mesh_lat = np.meshgrid([0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0])
    with patch("atmos_gl.lib.coastline._load_gshhg_land_union", return_value=land):
        mask = coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)

    assert mask[1, 1] == True  # noqa: E712 -- the original land cell
    # Every neighbour (including diagonal) of [1,1] is grown to True.
    assert mask[0, 0] == True and mask[0, 1] == True and mask[0, 2] == True  # noqa: E712
    assert mask[1, 0] == True and mask[1, 2] == True  # noqa: E712
    assert mask[2, 0] == True and mask[2, 1] == True and mask[2, 2] == True  # noqa: E712
    assert mask[3, 3] == False  # noqa: E712 -- too far to be grown into


def test_coastline_land_mask_dilate_false_skips_it():
    land = _square_land(5.0, 5.0, 15.0, 15.0)
    mesh_lon, mesh_lat = np.meshgrid([0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0])
    with patch("atmos_gl.lib.coastline._load_gshhg_land_union", return_value=land):
        mask = coastline_land_mask(
            mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0, dilate=False
        )

    assert mask[1, 1] == True  # noqa: E712
    assert mask[0, 0] == False and mask[0, 1] == False and mask[1, 0] == False  # noqa: E712 -- NOT grown


def test_coastline_land_mask_caches_the_union_per_bbox():
    land = _square_land(0.0, 0.0, 1.0, 1.0)
    mesh_lon, mesh_lat = np.meshgrid([0.5], [0.5])
    with patch(
        "atmos_gl.lib.coastline._load_gshhg_land_union", return_value=land
    ) as mock_load:
        coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)
        coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)

    mock_load.assert_called_once()


def test_coastline_land_mask_skips_intersection_for_the_global_bbox():
    """Every real caller passes the global bbox -- clipping against it would be a
    correct no-op, but a wasted intersection against a huge unioned geometry. Confirm
    the global path never calls .intersection() on the loaded union."""
    land = MagicMock(wraps=_square_land(0.0, 0.0, 10.0, 10.0))
    mesh_lon, mesh_lat = np.meshgrid([5.0], [5.0])
    with patch("atmos_gl.lib.coastline._load_gshhg_land_union", return_value=land):
        coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)

    land.intersection.assert_not_called()


def test_coastline_land_mask_clips_to_a_non_global_bbox():
    land = _square_land(-100.0, -100.0, 100.0, 100.0)
    mesh_lon, mesh_lat = np.meshgrid([5.0], [5.0])
    with patch("atmos_gl.lib.coastline._load_gshhg_land_union", return_value=land):
        mask = coastline_land_mask(mesh_lon, mesh_lat, 0.0, 0.0, 10.0, 10.0)

    assert mask.tolist() == [[True]]


def test_coastline_land_mask_returns_none_on_load_failure():
    mesh_lon, mesh_lat = np.meshgrid([0.0], [0.0])
    with patch(
        "atmos_gl.lib.coastline._load_gshhg_land_union",
        side_effect=RuntimeError("no network"),
    ):
        result = coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)

    assert result is None


# ---------------------------------------------------------------------------
# _load_gshhg_land_union
# ---------------------------------------------------------------------------

def test_load_gshhg_land_union_reads_from_disk_cache_if_present():
    cached = _square_land(1.0, 1.0, 2.0, 2.0)
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=b"fake-wkb-bytes")), \
         patch("shapely.from_wkb", return_value=cached) as mock_from_wkb:
        result = coastline_mod._load_gshhg_land_union()

    assert result is cached
    mock_from_wkb.assert_called_once_with(b"fake-wkb-bytes")


def test_load_gshhg_land_union_repairs_invalid_geometries_before_union():
    """GSHHG's raw polygons include some invalid by shapely's strict standards --
    unary_union raises a TopologyException outright without a make_valid() repair pass
    first (found live against the real 'h' tier data: 1 of 144,749 polygons)."""
    mock_reader = MagicMock()
    mock_reader.geometries.return_value = [
        _square_land(0.0, 0.0, 1.0, 1.0),
        _square_land(1.0, 1.0, 2.0, 2.0),
    ]

    with patch("os.path.exists", return_value=False), \
         patch(
             "atmos_gl.lib.coastline._download_gshhg_if_needed",
             return_value="/fake/GSHHS_h_L1.shp",
         ), \
         patch("cartopy.io.shapereader.Reader", return_value=mock_reader), \
         patch("builtins.open", mock_open()):
        result = coastline_mod._load_gshhg_land_union()

    # Both squares touch at one corner (0,0)-(1,1)-(2,2) -- the union covers both.
    assert result.contains(Point(0.5, 0.5))
    assert result.contains(Point(1.5, 1.5))


# ---------------------------------------------------------------------------
# nearest_fill_and_regrid_uv
# ---------------------------------------------------------------------------

def make_regrid_fn():
    """A fake regrid_for_lod: returns fixed new_lats/new_lons and the field
    UNCHANGED, so tests can inspect exactly what was passed in (post-nearest-fill)."""
    calls = []

    def regrid_fn(field, lats, lons, fill_value=np.nan, step_override=None):
        calls.append({"field": field.copy(), "step_override": step_override})
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), field

    return regrid_fn, calls


def test_regrids_both_u_and_v_with_the_given_step_override():
    regrid_fn, calls = make_regrid_fn()
    u = np.array([[1.0, 2.0]], dtype=np.float32)
    v = np.array([[3.0, 4.0]], dtype=np.float32)

    nearest_fill_and_regrid_uv(regrid_fn, u, v, [0.0], [0.0, 1.0], step_deg=0.08)

    assert len(calls) == 2
    assert calls[0]["step_override"] == 0.08
    assert calls[1]["step_override"] == 0.08


def test_nearest_fills_native_nan_before_regridding():
    regrid_fn, calls = make_regrid_fn()
    u = np.array([[1.0, np.nan, 1.0]], dtype=np.float32)
    v = np.array([[2.0, np.nan, 2.0]], dtype=np.float32)

    nearest_fill_and_regrid_uv(regrid_fn, u, v, [0.0], [0.0, 1.0, 2.0], step_deg=0.08)

    # regrid_fn must never see a NaN -- it was nearest-filled first.
    assert not np.isnan(calls[0]["field"]).any()
    assert not np.isnan(calls[1]["field"]).any()


def test_all_nan_native_is_left_unfilled():
    """distance_transform_edt needs SOME valid data to fill from -- an entirely-NaN
    field is left as-is (matches the original `bad.any() and not bad.all()` guard)."""
    regrid_fn, calls = make_regrid_fn()
    u = np.full((1, 2), np.nan, dtype=np.float32)
    v = np.full((1, 2), np.nan, dtype=np.float32)

    nearest_fill_and_regrid_uv(regrid_fn, u, v, [0.0], [0.0, 1.0], step_deg=0.08)

    assert np.isnan(calls[0]["field"]).all()
    assert np.isnan(calls[1]["field"]).all()


def test_does_not_mutate_the_callers_original_arrays():
    regrid_fn, _ = make_regrid_fn()
    u = np.array([[1.0, np.nan]], dtype=np.float32)
    v = np.array([[2.0, np.nan]], dtype=np.float32)
    u_original = u.copy()
    v_original = v.copy()

    nearest_fill_and_regrid_uv(regrid_fn, u, v, [0.0], [0.0, 1.0], step_deg=0.08)

    np.testing.assert_array_equal(u, u_original, err_msg="input u must be untouched", strict=True)
    np.testing.assert_array_equal(v, v_original, err_msg="input v must be untouched", strict=True)


def test_returns_new_lats_new_lons_and_both_regridded_fields():
    regrid_fn, _ = make_regrid_fn()
    u = np.array([[1.0, 2.0]], dtype=np.float32)
    v = np.array([[3.0, 4.0]], dtype=np.float32)

    new_lats, new_lons, out_u, out_v = nearest_fill_and_regrid_uv(
        regrid_fn, u, v, [0.0], [0.0, 1.0], step_deg=0.08
    )

    np.testing.assert_array_equal(new_lats, [0.0, 1.0])
    np.testing.assert_array_equal(new_lons, [0.0, 1.0])
    np.testing.assert_array_equal(out_u, u)
    np.testing.assert_array_equal(out_v, v)
