#!/usr/bin/env python3
"""FloodRiskHistoricalCollector: JRC Global River Flood Hazard Maps (100-year
return period), mosaicked once from 271 open-FTP tiles into a single global
raster and cached forever. Fetch-once/coverage-based, like
CamsEgg4BaselineCollector -- see test_greenhouse_gases_egg4_collector.py for the
sibling test pattern this mirrors.
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
from rasterio import Affine

from atmos_gl.collectors.flood_risk import FloodRiskHistoricalCollector
from atmos_gl.lib.flood_risk import (
    jrc_hazard_mosaic_cache_path,
    jrc_tile_extents_cache_path,
    load_jrc_hazard_mosaic,
    tile_dst_window,
)


def make_bare_historical_collector(settings=None, workdir="."):
    c = FloodRiskHistoricalCollector.__new__(FloodRiskHistoricalCollector)
    c.settings = settings or {}

    def fake_get_setting(section, key, default=None):
        if section == "common" and key == "workdir":
            return workdir
        return default

    c.config = MagicMock()
    c.config.get_setting.side_effect = fake_get_setting
    return c


def _write_tile_fixture(path, category, bounds):
    lon_min, lat_min, lon_max, lat_max = bounds
    transform = Affine(
        (lon_max - lon_min) / 2, 0.0, lon_min,
        0.0, -(lat_max - lat_min) / 2, lat_max,
    )
    values = np.full((2, 2), category, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=255,
    ) as dst:
        dst.write(values, 1)


_TILES = [
    {"id": 1, "name": "N70_W180", "bounds": (-180.0, 60.0, -170.0, 70.0)},
    {"id": 2, "name": "N60_W180", "bounds": (-180.0, 50.0, -170.0, 60.0)},
]


def test_collect_skips_entirely_when_mosaic_already_cached(tmp_path):
    c = make_bare_historical_collector(workdir=str(tmp_path))
    dest = jrc_hazard_mosaic_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b"already-cached")

    with patch("atmos_gl.collectors.flood_risk.ensure_jrc_tile_extents_cached") as mock_index:
        c.collect()

    mock_index.assert_not_called()
    assert open(dest, "rb").read() == b"already-cached"


def test_collect_returns_gracefully_when_tile_index_unavailable(tmp_path):
    c = make_bare_historical_collector(workdir=str(tmp_path))

    with patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_extents_cached",
        side_effect=Exception("network down"),
    ):
        c.collect()  # must not raise

    assert not os.path.exists(jrc_hazard_mosaic_cache_path(str(tmp_path)))


def test_collect_does_not_write_a_mosaic_when_only_some_tiles_are_cached_this_cycle(tmp_path):
    """A partial pass (one tile download failing) must not produce a half-complete
    mosaic -- it should log progress and let the next cycle retry the missing
    tile(s), since ensure_jrc_tile_cached()'s own skip-if-cached check makes that
    naturally resumable."""
    c = make_bare_historical_collector(workdir=str(tmp_path))
    good_tile_path = str(tmp_path / "good.tif")
    _write_tile_fixture(good_tile_path, 2, _TILES[0]["bounds"])

    def fake_ensure_tile(tile_id, tile_name):
        if tile_id == 1:
            return good_tile_path
        raise Exception("truncated download")

    with patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_extents_cached",
        return_value="fake-index-path",
    ), patch(
        "atmos_gl.collectors.flood_risk.load_jrc_tile_index", return_value=_TILES
    ), patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_cached", side_effect=fake_ensure_tile
    ):
        c.collect()

    assert not os.path.exists(jrc_hazard_mosaic_cache_path(str(tmp_path)))


def test_collect_builds_and_caches_the_mosaic_once_every_tile_succeeds(tmp_path):
    c = make_bare_historical_collector(workdir=str(tmp_path))
    tile_paths = {}
    for tile, category in zip(_TILES, (2, 4)):
        path = str(tmp_path / f"{tile['name']}.tif")
        _write_tile_fixture(path, category, tile["bounds"])
        tile_paths[tile["id"]] = path

    with patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_extents_cached",
        return_value="fake-index-path",
    ), patch(
        "atmos_gl.collectors.flood_risk.load_jrc_tile_index", return_value=_TILES
    ), patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_cached",
        side_effect=lambda tile_id, tile_name: tile_paths[tile_id],
    ):
        c.collect()

    dest = jrc_hazard_mosaic_cache_path(str(tmp_path))
    assert os.path.exists(dest)
    band, lat, lon = load_jrc_hazard_mosaic(dest)

    row0, row1, col0, col1 = tile_dst_window(_TILES[0]["bounds"])
    assert (band[row0:row1, col0:col1] == 2).all()
    row0, row1, col0, col1 = tile_dst_window(_TILES[1]["bounds"])
    assert (band[row0:row1, col0:col1] == 4).all()
    # Untouched land (no tile covers it) stays at the mosaic's own default fill.
    assert band[-1, -1] == 0


def test_collect_stops_downloading_new_tiles_once_the_per_cycle_budget_is_reached(tmp_path, monkeypatch):
    """Regression guard for the head-of-line-blocking bug found live on prod: a
    single collect() call downloading every remaining tile can run long enough to
    push the data_collector heartbeat past the Data Status page's dead threshold
    (see the budget constant's own comment in collectors/flood_risk.py). With the
    budget set to 1, a 2-tile index must download only the first tile this cycle
    and return without building the mosaic, leaving the second tile for next time."""
    c = make_bare_historical_collector(workdir=str(tmp_path))
    monkeypatch.setattr(
        "atmos_gl.collectors.flood_risk.FloodRiskHistoricalCollector"
        "._MAX_NEW_TILE_DOWNLOADS_PER_CYCLE",
        1,
    )
    good_tile_path = str(tmp_path / "good.tif")
    _write_tile_fixture(good_tile_path, 2, _TILES[0]["bounds"])
    calls = []

    def fake_ensure_tile(tile_id, tile_name):
        calls.append(tile_id)
        return good_tile_path

    with patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_extents_cached",
        return_value="fake-index-path",
    ), patch(
        "atmos_gl.collectors.flood_risk.load_jrc_tile_index", return_value=_TILES
    ), patch(
        "atmos_gl.collectors.flood_risk.jrc_tile_cache_path",
        return_value=str(tmp_path / "not-cached-yet.tif"),
    ), patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_cached", side_effect=fake_ensure_tile
    ):
        c.collect()

    assert calls == [1]  # only the first (budget=1) tile was fetched
    assert not os.path.exists(jrc_hazard_mosaic_cache_path(str(tmp_path)))


def test_collect_does_not_count_already_cached_tiles_against_the_download_budget(tmp_path, monkeypatch):
    """Resuming a mostly-complete mosaic must not stall forever: tiles already on
    disk are free re-reads (no network hit), so only genuinely NEW downloads count
    against the per-cycle budget -- with budget=1 and tile 1 already cached, tile 2
    (the only new one) still gets fetched and the mosaic completes this cycle."""
    c = make_bare_historical_collector(workdir=str(tmp_path))
    monkeypatch.setattr(
        "atmos_gl.collectors.flood_risk.FloodRiskHistoricalCollector"
        "._MAX_NEW_TILE_DOWNLOADS_PER_CYCLE",
        1,
    )
    tile_paths = {}
    for tile, category in zip(_TILES, (2, 4)):
        path = str(tmp_path / f"{tile['name']}.tif")
        _write_tile_fixture(path, category, tile["bounds"])
        tile_paths[tile["id"]] = path

    def fake_jrc_tile_cache_path(tile_id, tile_name):
        # Tile 1 looks already-cached on disk; tile 2 does not.
        return tile_paths[1] if tile_id == 1 else str(tmp_path / "not-cached-yet.tif")

    with patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_extents_cached",
        return_value="fake-index-path",
    ), patch(
        "atmos_gl.collectors.flood_risk.load_jrc_tile_index", return_value=_TILES
    ), patch(
        "atmos_gl.collectors.flood_risk.jrc_tile_cache_path",
        side_effect=fake_jrc_tile_cache_path,
    ), patch(
        "atmos_gl.collectors.flood_risk.ensure_jrc_tile_cached",
        side_effect=lambda tile_id, tile_name: tile_paths[tile_id],
    ):
        c.collect()

    assert os.path.exists(jrc_hazard_mosaic_cache_path(str(tmp_path)))


def test_source_url_is_the_hardcoded_jrc_ftp_base_not_a_config_datasource(tmp_path):
    c = make_bare_historical_collector(workdir=str(tmp_path))
    assert c.source_url() == "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"


def test_data_status_is_zero_before_the_tile_index_is_ever_fetched(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = make_bare_historical_collector(workdir=str(tmp_path))
    c.process_status_adapter = MagicMock()
    c.process_status_adapter.get_process_status.return_value = None

    status = c.data_status()

    assert status["percent"] == 0.0


def test_data_status_reflects_tile_download_progress_before_the_mosaic_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = make_bare_historical_collector(workdir=str(tmp_path))
    c.process_status_adapter = MagicMock()
    c.process_status_adapter.get_process_status.return_value = None

    index_path = jrc_tile_extents_cache_path()
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    import json

    with open(index_path, "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": t["id"], "name": t["name"]},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [t["bounds"][0], t["bounds"][3]], [t["bounds"][2], t["bounds"][3]],
                                [t["bounds"][2], t["bounds"][1]], [t["bounds"][0], t["bounds"][1]],
                                [t["bounds"][0], t["bounds"][3]],
                            ]],
                        },
                    }
                    for t in _TILES
                ],
            },
            f,
        )

    from atmos_gl.lib.flood_risk import jrc_tile_cache_path

    cached_tile = jrc_tile_cache_path(_TILES[0]["id"], _TILES[0]["name"])
    os.makedirs(os.path.dirname(cached_tile), exist_ok=True)
    with open(cached_tile, "wb") as f:
        f.write(b"fake")

    status = c.data_status()

    assert status["percent"] == 50.0


def test_data_status_is_100_once_the_mosaic_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = make_bare_historical_collector(workdir=str(tmp_path))
    c.process_status_adapter = MagicMock()
    c.process_status_adapter.get_process_status.return_value = None

    dest = jrc_hazard_mosaic_cache_path(str(tmp_path))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b"mosaic")

    status = c.data_status()

    assert status["percent"] == 100.0
