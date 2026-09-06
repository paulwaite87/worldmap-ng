#!/usr/bin/env python3
"""Tests for lib/vegetation_mask.py -- the burnable-vegetation mask behind the
Fire Risk layer's land+vegetation masking (issue #390). Pure mask-logic and
Zenodo-response-parsing tests, no live network calls; mirrors the synthetic-
GeoTIFF-fixture pattern already used by test_flood_risk_historical_collector.py
and test_flood_risk_live_collector.py for the same "reproject a categorical raster
onto a destination grid" shape.
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from atmos_gl.lib.vegetation_mask import (
    BURNABLE_IGBP_CLASSES,
    VegetationMaskCache,
    ZENODO_VERSIONS_LATEST_URL,
    _remap_igbp_to_burnable,
    burnable_vegetation_mask,
    cached_version_id,
    download_landcover_geotiff,
    fetch_latest_zenodo_version,
    find_landcover_asset,
    save_cached_version_id,
    vegetation_mask_geotiff_cache_path,
    vegetation_mask_version_cache_path,
)


# ---- IGBP burnable classification -------------------------------------------


def test_burnable_classes_match_the_agreed_igbp_split():
    """Locks the exact split agreed in issue #390: forest/shrub/savanna/grassland/
    wetland/cropland types are burnable; urban, snow/ice, barren, and water are not."""
    assert BURNABLE_IGBP_CLASSES == frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14})


def test_remap_igbp_to_burnable_classifies_every_class_correctly():
    source = np.array([[1, 11, 13], [14, 16, 17]], dtype=np.uint8)
    result = _remap_igbp_to_burnable(source, None)
    assert result.tolist() == [[1, 1, 0], [1, 0, 0]]


# ---- find_landcover_asset ----------------------------------------------------


def _t1_file(start, end, extra_id=""):
    return {
        "key": f"lc_mcd12q1v061.t1_c_500m_s_{start}_{end}_go_epsg.4326_v20230818.tif{extra_id}",
        "size": 132000000,
        "links": {"self": f"https://zenodo.org/api/records/8367523/files/{start}_{end}.tif/content"},
    }


def test_find_landcover_asset_picks_the_latest_end_date():
    version_json = {
        "files": [
            _t1_file("20190101", "20191231"),
            _t1_file("20210101", "20211231"),
            _t1_file("20050101", "20051231"),
            # A different band ("t2"/"qc") must never be mistaken for the t1 (IGBP) file.
            {"key": "lc_mcd12q1v061.qc_c_500m_s_20210101_20211231_go_epsg.4326_v20230818.tif",
             "links": {"self": "https://example.test/qc.tif"}},
        ]
    }
    asset = find_landcover_asset(version_json)
    assert asset["key"] == "lc_mcd12q1v061.t1_c_500m_s_20210101_20211231_go_epsg.4326_v20230818.tif"


def test_find_landcover_asset_returns_none_when_no_t1_file_present():
    version_json = {"files": [{"key": "some_other_product.tif", "links": {"self": "x"}}]}
    assert find_landcover_asset(version_json) is None


def test_find_landcover_asset_handles_an_empty_files_list():
    assert find_landcover_asset({"files": []}) is None
    assert find_landcover_asset({}) is None


# ---- fetch_latest_zenodo_version ---------------------------------------------


def test_fetch_latest_zenodo_version_gets_the_versions_latest_endpoint():
    fake_response = MagicMock()
    fake_response.json.return_value = {"id": 8367523, "files": []}
    with patch("requests.get", return_value=fake_response) as mock_get:
        result = fetch_latest_zenodo_version()

    mock_get.assert_called_once_with(ZENODO_VERSIONS_LATEST_URL, timeout=15)
    fake_response.raise_for_status.assert_called_once()
    assert result == {"id": 8367523, "files": []}


def test_fetch_latest_zenodo_version_raises_on_http_error():
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = Exception("500 server error")
    with patch("requests.get", return_value=fake_response):
        with pytest.raises(Exception):
            fetch_latest_zenodo_version()


# ---- download_landcover_geotiff ----------------------------------------------


def test_download_landcover_geotiff_writes_the_downloaded_bytes(tmp_path):
    dest = str(tmp_path / "sub" / "landcover.tif")
    with patch("atmos_gl.lib.gfs.download_whole", return_value=b"fake-tif-bytes") as mock_dl:
        download_landcover_geotiff("https://example.test/x.tif", dest)

    mock_dl.assert_called_once_with("https://example.test/x.tif", timeout=300)
    assert os.path.exists(dest)
    with open(dest, "rb") as f:
        assert f.read() == b"fake-tif-bytes"


# ---- version cache sidecar ----------------------------------------------------


def test_cached_version_id_is_none_when_nothing_cached_yet(tmp_path):
    assert cached_version_id(str(tmp_path)) is None


def test_save_and_read_cached_version_id_round_trips(tmp_path):
    save_cached_version_id(str(tmp_path), 8367523)
    assert cached_version_id(str(tmp_path)) == 8367523


def test_cached_version_id_is_none_when_sidecar_is_corrupt(tmp_path):
    path = vegetation_mask_version_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("not json")
    assert cached_version_id(str(tmp_path)) is None


def test_cache_paths_live_under_workdir_data():
    assert vegetation_mask_geotiff_cache_path("/wd") == "/wd/data/vegetation_mask_cache_landcover.tif"
    assert vegetation_mask_version_cache_path("/wd") == "/wd/data/vegetation_mask_cache_version.json"


# ---- burnable_vegetation_mask / VegetationMaskCache --------------------------


def _write_landcover_fixture(path, values, bounds):
    """A tiny synthetic 2x2 IGBP-classified GeoTIFF -- stands in for the real
    ~130MB global mosaic, mirroring test_flood_risk_historical_collector.py's
    _write_tile_fixture exactly (north-up: row 0 = lat_max)."""
    lon_min, lat_min, lon_max, lat_max = bounds
    height, width = values.shape
    transform = Affine(
        (lon_max - lon_min) / width, 0.0, lon_min,
        0.0, -(lat_max - lat_min) / height, lat_max,
    )
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(values, 1)


_BOUNDS = (0.0, 0.0, 20.0, 20.0)
# row 0 (north, lat~15): [Evergreen Needleleaf Forest (1, burnable), Water (17, not)]
# row 1 (south, lat~5):  [Barren (16, not),                          Grasslands (10, burnable)]
_FIXTURE_VALUES = np.array([[1, 17], [16, 10]], dtype=np.uint8)


def test_burnable_vegetation_mask_is_none_when_raster_not_yet_cached(tmp_path):
    assert burnable_vegetation_mask([10.0, 0.0], [5.0, 15.0], str(tmp_path)) is None


def test_burnable_vegetation_mask_classifies_correctly_for_a_descending_native_grid(tmp_path):
    """Native fieldstore grids are north-first (descending) -- the common case,
    and the orientation _reproject_categorical_max was already built for."""
    path = vegetation_mask_geotiff_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_landcover_fixture(path, _FIXTURE_VALUES, _BOUNDS)

    mask = burnable_vegetation_mask([15.0, 5.0], [5.0, 15.0], str(tmp_path))

    assert mask.tolist() == [[True, False], [False, True]]


def test_burnable_vegetation_mask_classifies_correctly_for_an_ascending_lod_grid(tmp_path):
    """regrid_for_lod's LOD grid is ascending (south-first) -- the opposite
    convention from the native grid above. The returned mask's row order must
    track the CALLER's lat order, not silently stay north-first -- this is the
    orientation bug this function specifically guards against."""
    path = vegetation_mask_geotiff_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_landcover_fixture(path, _FIXTURE_VALUES, _BOUNDS)

    mask = burnable_vegetation_mask([5.0, 15.0], [5.0, 15.0], str(tmp_path))

    # row 0 is now lat=5 (south) -> [Barren, Grasslands] -> [False, True]
    # row 1 is now lat=15 (north) -> [Forest, Water]      -> [True, False]
    assert mask.tolist() == [[False, True], [True, False]]


def test_burnable_vegetation_mask_returns_none_on_a_corrupt_raster(tmp_path):
    path = vegetation_mask_geotiff_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("not a real geotiff")

    assert burnable_vegetation_mask([10.0, 0.0], [5.0, 15.0], str(tmp_path)) is None


def test_vegetation_mask_cache_caches_per_key(tmp_path):
    path = vegetation_mask_geotiff_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_landcover_fixture(path, _FIXTURE_VALUES, _BOUNDS)

    cache = VegetationMaskCache("Test", str(tmp_path))
    first = cache.get([15.0, 5.0], [5.0, 15.0], "key-a")
    with patch(
        "atmos_gl.lib.vegetation_mask.burnable_vegetation_mask"
    ) as mock_build:
        second = cache.get([15.0, 5.0], [5.0, 15.0], "key-a")
        mock_build.assert_not_called()

    assert (first == second).all()


def test_vegetation_mask_cache_treats_different_keys_independently(tmp_path):
    path = vegetation_mask_geotiff_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_landcover_fixture(path, _FIXTURE_VALUES, _BOUNDS)

    cache = VegetationMaskCache("Test", str(tmp_path))
    cache.get([15.0, 5.0], [5.0, 15.0], "key-a")
    with patch(
        "atmos_gl.lib.vegetation_mask.burnable_vegetation_mask", return_value=None
    ) as mock_build:
        cache.get([15.0, 5.0], [5.0, 15.0], "key-b")
        mock_build.assert_called_once()
