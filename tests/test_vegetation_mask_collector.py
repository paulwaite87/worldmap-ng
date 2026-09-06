#!/usr/bin/env python3
"""Tests for VegetationMaskCollector (collectors/vegetation_mask.py) -- the
periodic Zenodo-version-check collector behind the Fire Risk layer's
burnable-vegetation mask (issue #390). Mirrors test_flood_risk_live_collector.py's
"has_new_data() does the real check, collect() reuses it" test shape, with mocked
HTTP throughout -- no live network calls.
"""
import os
from unittest.mock import MagicMock, patch

from atmos_gl.collectors.vegetation_mask import VegetationMaskCollector
from atmos_gl.lib.vegetation_mask import (
    ZENODO_RECORD_HTML_URL,
    cached_version_id,
    vegetation_mask_geotiff_cache_path,
)


def make_bare_vegetation_mask_collector(settings=None, workdir="."):
    c = VegetationMaskCollector.__new__(VegetationMaskCollector)
    c.settings = settings or {}

    def fake_get_setting(section, key, default=None):
        if section == "common" and key == "workdir":
            return workdir
        return default

    c.config = MagicMock()
    c.config.get_setting.side_effect = fake_get_setting
    return c


_VERSION_A = {
    "id": 8367523,
    "files": [
        {
            "key": "lc_mcd12q1v061.t1_c_500m_s_20210101_20211231_go_epsg.4326_v20230818.tif",
            "links": {"self": "https://example.test/2021.tif"},
        }
    ],
}
_VERSION_B = {
    "id": 9999999,
    "files": [
        {
            "key": "lc_mcd12q1v061.t1_c_500m_s_20220101_20221231_go_epsg.4326_v20240101.tif",
            "links": {"self": "https://example.test/2022.tif"},
        }
    ],
}


# ---- source_url ---------------------------------------------------------------


def test_source_url_is_the_hardcoded_zenodo_endpoint_not_a_config_datasource():
    c = make_bare_vegetation_mask_collector()
    assert c.source_url() == ZENODO_RECORD_HTML_URL


# ---- channel/settings sharing with Fires -------------------------------------


def test_shares_fires_channel_and_settings_section():
    assert VegetationMaskCollector.channel_key == "fires"
    assert VegetationMaskCollector.settings_section == "fires"
    assert VegetationMaskCollector.section == "vegetation_mask"


# ---- has_new_data ---------------------------------------------------------------


def test_has_new_data_is_false_when_version_check_fails(tmp_path):
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.vegetation_mask.fetch_latest_zenodo_version",
        side_effect=Exception("network down"),
    ):
        assert c.has_new_data() is False

    assert c._latest_version is None


def test_has_new_data_is_true_when_nothing_cached_yet(tmp_path):
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.vegetation_mask.fetch_latest_zenodo_version",
        return_value=_VERSION_A,
    ):
        assert c.has_new_data() is True

    assert c._latest_version == _VERSION_A


def test_has_new_data_is_false_when_version_id_matches_cached(tmp_path):
    from atmos_gl.lib.vegetation_mask import save_cached_version_id

    save_cached_version_id(str(tmp_path), _VERSION_A["id"])
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.vegetation_mask.fetch_latest_zenodo_version",
        return_value=_VERSION_A,
    ):
        assert c.has_new_data() is False


def test_has_new_data_is_true_when_a_newer_version_is_published(tmp_path):
    from atmos_gl.lib.vegetation_mask import save_cached_version_id

    save_cached_version_id(str(tmp_path), _VERSION_A["id"])
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.vegetation_mask.fetch_latest_zenodo_version",
        return_value=_VERSION_B,
    ):
        assert c.has_new_data() is True


# ---- collect ---------------------------------------------------------------


def test_collect_downloads_and_caches_the_new_version(tmp_path):
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))
    c._latest_version = _VERSION_A

    with patch(
        "atmos_gl.collectors.vegetation_mask.download_landcover_geotiff"
    ) as mock_dl:
        c.collect()

    mock_dl.assert_called_once_with(
        "https://example.test/2021.tif", vegetation_mask_geotiff_cache_path(str(tmp_path))
    )
    assert cached_version_id(str(tmp_path)) == _VERSION_A["id"]


def test_collect_refetches_the_version_when_has_new_data_was_not_called_first(tmp_path):
    """collect() must not assume has_new_data() always ran immediately before it
    (true in production via EventFeedDriver, but not necessarily of a direct
    call, e.g. in a test) -- it should refetch rather than silently do nothing."""
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))
    assert not hasattr(c, "_latest_version")

    with patch(
        "atmos_gl.collectors.vegetation_mask.fetch_latest_zenodo_version",
        return_value=_VERSION_A,
    ), patch("atmos_gl.collectors.vegetation_mask.download_landcover_geotiff") as mock_dl:
        c.collect()

    mock_dl.assert_called_once()
    assert cached_version_id(str(tmp_path)) == _VERSION_A["id"]


def test_collect_skips_gracefully_when_version_check_fails_and_was_not_cached(tmp_path):
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.vegetation_mask.fetch_latest_zenodo_version",
        side_effect=Exception("network down"),
    ), patch("atmos_gl.collectors.vegetation_mask.download_landcover_geotiff") as mock_dl:
        c.collect()  # must not raise

    mock_dl.assert_not_called()
    assert not os.path.exists(vegetation_mask_geotiff_cache_path(str(tmp_path)))


def test_collect_skips_when_no_t1_asset_found_in_the_version(tmp_path):
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))
    c._latest_version = {"id": 123, "files": [{"key": "unrelated.tif", "links": {"self": "x"}}]}

    with patch("atmos_gl.collectors.vegetation_mask.download_landcover_geotiff") as mock_dl:
        c.collect()  # must not raise

    mock_dl.assert_not_called()
    assert cached_version_id(str(tmp_path)) is None


def test_collect_does_not_advance_the_cached_version_when_download_fails(tmp_path):
    c = make_bare_vegetation_mask_collector(workdir=str(tmp_path))
    c._latest_version = _VERSION_A

    with patch(
        "atmos_gl.collectors.vegetation_mask.download_landcover_geotiff",
        side_effect=Exception("network down"),
    ):
        c.collect()  # must not raise

    assert cached_version_id(str(tmp_path)) is None
