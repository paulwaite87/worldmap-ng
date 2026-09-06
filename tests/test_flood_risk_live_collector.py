#!/usr/bin/env python3
"""FloodRiskLiveCollector: NASA LANCE MODIS flood product ("Observed Current
Inundation"), rebuilt from cached tiles every cycle has_new_data() finds a
changed or newly-expired tile. A CollectorBase subclass (file-cache family, like
its Historical sibling) -- see test_flood_risk_historical_collector.py for the
sibling test pattern this mirrors, and collectors/flood_risk.py's module
docstring for why this replaced the original GloFAS-forecast design.
"""
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
from rasterio import Affine

from atmos_gl.collectors.flood_risk import FloodRiskLiveCollector
from atmos_gl.lib.flood_risk import (
    load_jrc_hazard_mosaic,
    modis_flood_mosaic_cache_path,
    modis_flood_tile_bounds,
    modis_flood_tile_cache_path,
    tile_dst_window,
)


def make_bare_live_collector(settings=None, workdir="."):
    c = FloodRiskLiveCollector.__new__(FloodRiskLiveCollector)
    c.settings = settings or {}

    def fake_get_setting(section, key, default=None):
        if section == "common" and key == "workdir":
            return workdir
        return default

    c.config = MagicMock()
    c.config.get_setting.side_effect = fake_get_setting
    return c


def _write_tile_fixture(path, values, bounds):
    lon_min, lat_min, lon_max, lat_max = bounds
    height, width = values.shape
    transform = Affine(
        (lon_max - lon_min) / width, 0.0, lon_min,
        0.0, -(lat_max - lat_min) / height, lat_max,
    )
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=255,
    ) as dst:
        dst.write(values, 1)


def _cache_tile(h, v, value, filename):
    """Write a fully-cached tile (GeoTIFF + .name sidecar) directly, bypassing
    ensure_modis_flood_tile_cached, for tests that need a pre-existing cache
    without exercising the download path. Always includes an adjacent "surface
    water" (1) pixel alongside `value` -- irrelevant unless value is
    MODIS_FLOOD_VALUE (3), in which case resample_modis_flood_tile_onto_grid's
    water-adjacency filter (see MODIS_FLOOD_WATER_ADJACENCY_PX) would otherwise
    always drop it."""
    tile_path = modis_flood_tile_cache_path(h, v)
    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
    _write_tile_fixture(
        tile_path, np.array([[value, 1]], dtype=np.uint8), modis_flood_tile_bounds(h, v)
    )
    with open(tile_path + ".name", "w") as f:
        f.write(filename)
    return tile_path


_TILE_A = {
    "h": 19, "v": 6, "filename": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.111.tif",
    "download_url": "https://example.test/h19v06.tif",
}
_TILE_B = {
    "h": 20, "v": 6, "filename": "MCDWD_L3_F1C_NRT.A2026242.h20v06.061.111.tif",
    "download_url": "https://example.test/h20v06.tif",
}


# ---- source_url ---------------------------------------------------------------


def test_source_url_is_the_hardcoded_lance_endpoint_not_a_config_datasource():
    c = make_bare_live_collector()
    assert c.source_url() == "https://nrt3.modaps.eosdis.nasa.gov"


# ---- has_new_data ---------------------------------------------------------------


def test_has_new_data_is_false_when_no_token_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    c = make_bare_live_collector(workdir=str(tmp_path))

    with patch("atmos_gl.collectors.flood_risk.fetch_modis_flood_listing") as mock_fetch:
        assert c.has_new_data() is False

    mock_fetch.assert_not_called()


def test_has_new_data_is_false_when_listing_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.flood_risk.fetch_modis_flood_listing",
        side_effect=Exception("network down"),
    ):
        assert c.has_new_data() is False


def test_has_new_data_is_true_when_a_tile_is_new(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.flood_risk.fetch_modis_flood_listing", return_value=[_TILE_A]
    ):
        assert c.has_new_data() is True

    assert c._listing == [_TILE_A]


def test_has_new_data_is_false_when_every_tile_is_already_current(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))
    _cache_tile(_TILE_A["h"], _TILE_A["v"], 0, _TILE_A["filename"])

    with patch(
        "atmos_gl.collectors.flood_risk.fetch_modis_flood_listing", return_value=[_TILE_A]
    ):
        assert c.has_new_data() is False


def test_has_new_data_is_true_when_pruning_removed_a_stale_tile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))
    from atmos_gl.lib.flood_risk import MODIS_FLOOD_STALE_S

    tile_path = _cache_tile(_TILE_A["h"], _TILE_A["v"], 0, _TILE_A["filename"])
    old = time.time() - (MODIS_FLOOD_STALE_S + 3600)
    os.utime(tile_path, (old, old))
    os.utime(tile_path + ".name", (old, old))

    with patch("atmos_gl.collectors.flood_risk.fetch_modis_flood_listing", return_value=[]):
        assert c.has_new_data() is True

    assert not os.path.exists(tile_path)


# ---- collect ---------------------------------------------------------------


def test_collect_returns_gracefully_without_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    c = make_bare_live_collector(workdir=str(tmp_path))

    c.collect()  # must not raise

    assert not os.path.exists(modis_flood_mosaic_cache_path(str(tmp_path)))


def test_collect_downloads_changed_tiles_and_rebuilds_the_mosaic(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))
    c._listing = [_TILE_A, _TILE_B]

    fixture_bytes = {}
    # 3 = Flood (with an adjacent water pixel -- see MODIS_FLOOD_WATER_ADJACENCY_PX),
    # 0 = no water
    for tile, pixel_row in ((_TILE_A, [3, 1]), (_TILE_B, [0, 0])):
        p = tmp_path / f"src_{tile['h']}_{tile['v']}.tif"
        _write_tile_fixture(
            str(p), np.array([pixel_row], dtype=np.uint8),
            modis_flood_tile_bounds(tile["h"], tile["v"]),
        )
        fixture_bytes[tile["download_url"]] = p.read_bytes()

    def fake_download(url, headers=None):
        return fixture_bytes[url]

    with patch("atmos_gl.lib.gfs.download_whole", side_effect=fake_download):
        c.collect()

    dest = modis_flood_mosaic_cache_path(str(tmp_path))
    assert os.path.exists(dest)
    band, _lat, _lon = load_jrc_hazard_mosaic(dest)
    row0, row1, col0, col1 = tile_dst_window(modis_flood_tile_bounds(_TILE_A["h"], _TILE_A["v"]))
    col_mid = col0 + (col1 - col0) // 2
    # left half overlaps the source's Flood pixel -> binarized to 1; right half
    # overlaps the adjacent water pixel itself, which isn't flood -> 0
    assert (band[row0:row1, col0:col_mid] == 1).all()
    assert (band[row0:row1, col_mid:col1] == 0).all()
    row0, row1, col0, col1 = tile_dst_window(modis_flood_tile_bounds(_TILE_B["h"], _TILE_B["v"]))
    assert (band[row0:row1, col0:col1] == 0).all()


def test_collect_skips_a_tile_thats_already_current_but_still_mosaics_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))
    c._listing = [_TILE_A]
    _cache_tile(_TILE_A["h"], _TILE_A["v"], 3, _TILE_A["filename"])

    with patch("atmos_gl.lib.gfs.download_whole") as mock_download:
        c.collect()

    mock_download.assert_not_called()
    dest = modis_flood_mosaic_cache_path(str(tmp_path))
    assert os.path.exists(dest)
    band, _lat, _lon = load_jrc_hazard_mosaic(dest)
    row0, row1, col0, col1 = tile_dst_window(modis_flood_tile_bounds(_TILE_A["h"], _TILE_A["v"]))
    col_mid = col0 + (col1 - col0) // 2
    assert (band[row0:row1, col0:col_mid] == 1).all()  # Flood pixel half


def test_collect_keeps_a_tiles_previous_content_when_its_refresh_fails(tmp_path, monkeypatch):
    """A tile that fails to download this cycle must still contribute its
    last-known-good cached content to the mosaic, not be dropped -- same
    resilience JRC's per-tile cache already provides for Historical mode."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))
    stale_filename = "MCDWD_L3_F1C_NRT.A2026241.h19v06.061.000.tif"
    c._listing = [{**_TILE_A, "filename": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.999.tif"}]
    _cache_tile(_TILE_A["h"], _TILE_A["v"], 3, stale_filename)

    with patch("atmos_gl.lib.gfs.download_whole", side_effect=Exception("network down")):
        c.collect()

    dest = modis_flood_mosaic_cache_path(str(tmp_path))
    assert os.path.exists(dest)
    band, _lat, _lon = load_jrc_hazard_mosaic(dest)
    row0, row1, col0, col1 = tile_dst_window(modis_flood_tile_bounds(_TILE_A["h"], _TILE_A["v"]))
    col_mid = col0 + (col1 - col0) // 2
    assert (band[row0:row1, col0:col_mid] == 1).all()  # still the OLD (flood=3) content


def test_collect_writes_no_mosaic_when_nothing_has_ever_been_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))
    c._listing = [_TILE_A]

    with patch("atmos_gl.lib.gfs.download_whole", side_effect=Exception("network down")):
        c.collect()

    assert not os.path.exists(modis_flood_mosaic_cache_path(str(tmp_path)))


def test_collect_refetches_the_listing_when_has_new_data_was_not_called_first(tmp_path, monkeypatch):
    """collect() must not assume has_new_data() always ran immediately before it
    (true in production via EventFeedDriver, but not necessarily of a direct
    call, e.g. in a test) -- it should refetch rather than silently do nothing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    c = make_bare_live_collector(workdir=str(tmp_path))
    assert not hasattr(c, "_listing")

    tif_path = tmp_path / "src.tif"
    _write_tile_fixture(
        str(tif_path), np.array([[3, 1]], dtype=np.uint8),
        modis_flood_tile_bounds(_TILE_A["h"], _TILE_A["v"]),
    )

    with patch(
        "atmos_gl.collectors.flood_risk.fetch_modis_flood_listing", return_value=[_TILE_A]
    ), patch("atmos_gl.lib.gfs.download_whole", return_value=tif_path.read_bytes()):
        c.collect()

    assert os.path.exists(modis_flood_mosaic_cache_path(str(tmp_path)))
