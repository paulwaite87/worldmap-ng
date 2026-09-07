#!/usr/bin/env python3
"""FloodRiskUpdater.run() orchestration: both Live (MODIS-observed) and
Historical (JRC hazard) are single-shot cached-mosaic renders now (Live used to
be a per-forecast-hour GloFAS series -- see collectors/flood_risk.py's module
docstring), both rendered every cycle regardless of the configured mode, but only
the currently-configured mode's output is published to the stable base filename.
_render_live/_render_historical are mocked throughout for the orchestration
tests: this seam tests what gets rendered/published, not rendering internals
(covered separately below).
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np

from atmos_gl.lib.flood_risk import (
    jrc_hazard_mosaic_cache_path,
    modis_flood_mosaic_cache_path,
    save_jrc_hazard_mosaic,
)
from atmos_gl.tasks.flood_risk import (
    _HISTORICAL_ENCODE_DOMAIN,
    _LIVE_ENCODE_DOMAIN,
    FloodRiskUpdater,
)


def make_bare_flood_risk_updater(mode, workdir, output_path):
    u = FloodRiskUpdater.__new__(FloodRiskUpdater)
    u.mode = mode
    u.workdir = workdir
    u.section = "flood_risk"
    u.output_path = output_path
    u.settings = {"mode": mode}
    u.common = {}
    u._render_live = MagicMock(return_value=None)
    u._render_historical = MagicMock(return_value=None)
    u._publish_variant = MagicMock()
    return u


def test_run_always_renders_live_regardless_of_configured_mode(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = make_bare_flood_risk_updater("historical", str(tmp_path), str(out_path))

    u.run()

    u._render_live.assert_called_once()


def test_run_always_renders_historical_regardless_of_configured_mode(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = make_bare_flood_risk_updater("live", str(tmp_path), str(out_path))

    u.run()

    u._render_historical.assert_called_once()


def test_run_publishes_the_live_variant_when_mode_is_live_and_it_exists(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = make_bare_flood_risk_updater("live", str(tmp_path), str(out_path))
    live_variant = str(tmp_path / "data" / "flood_risk_live.png")
    u._render_live.return_value = live_variant

    u.run()

    u._publish_variant.assert_called_once_with(live_variant)


def test_run_publishes_nothing_when_mode_is_live_but_no_live_render_exists_yet(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = make_bare_flood_risk_updater("live", str(tmp_path), str(out_path))
    u._render_live.return_value = None

    u.run()

    u._publish_variant.assert_not_called()


def test_run_publishes_the_historical_variant_when_mode_is_historical(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = make_bare_flood_risk_updater("historical", str(tmp_path), str(out_path))
    historical_variant = str(tmp_path / "data" / "flood_risk_historical.png")
    u._render_historical.return_value = historical_variant

    u.run()

    u._publish_variant.assert_called_once_with(historical_variant)


def test_run_publishes_nothing_when_mode_is_historical_but_mosaic_not_cached_yet(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = make_bare_flood_risk_updater("historical", str(tmp_path), str(out_path))
    u._render_historical.return_value = None  # mosaic not cached yet

    u.run()

    u._publish_variant.assert_not_called()


def test_run_accepts_max_hours_as_a_no_op(tmp_path):
    """Every TASK_CLASSES entry's run() is called the same way by layer_builder,
    including single-shot layers -- see MarkerUpdater.run's identical
    convention."""
    out_path = tmp_path / "data" / "flood_risk.png"
    u = make_bare_flood_risk_updater("live", str(tmp_path), str(out_path))

    u.run(max_hours=3)  # must not raise


# ---- direct rendering tests (not orchestration) --------------------------------


def _bare_updater_for_rendering(workdir, output_path):
    u = FloodRiskUpdater.__new__(FloodRiskUpdater)
    u.workdir = workdir
    u.section = "flood_risk"
    u.output_path = output_path
    u.settings = {}
    # Default: no land mask applied (mirrors FireWeatherUpdater's bare-updater
    # test fixture) -- tests that care about masking set a real return value.
    u._land_mask_cache = MagicMock()
    u._land_mask_cache.get.return_value = None
    return u


def test_render_live_writes_a_binary_flood_texture_when_mosaic_is_cached(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))
    mosaic_path = modis_flood_mosaic_cache_path(str(tmp_path))
    band = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    save_jrc_hazard_mosaic(mosaic_path, band, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    out = u._render_live()

    assert out == str(tmp_path / "data" / "flood_risk_live.png")
    assert os.path.exists(out)
    assert os.path.exists(out + ".sig")
    assert _LIVE_ENCODE_DOMAIN == (0.0, 1.0)


def test_render_live_returns_none_when_mosaic_not_cached(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))

    assert u._render_live() is None


def test_render_live_skips_re_render_when_already_fresh(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))
    mosaic_path = modis_flood_mosaic_cache_path(str(tmp_path))
    band = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    save_jrc_hazard_mosaic(mosaic_path, band, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    first = u._render_live()
    first_mtime = os.path.getmtime(first)

    second = u._render_live()

    assert second == first
    assert os.path.getmtime(second) == first_mtime


# ---- _render_live's land mask (issue: Marlborough Sounds false-positive) -------


def test_render_live_zeroes_non_land_cells_when_a_land_mask_is_available(tmp_path):
    """resample_modis_flood_tile_onto_grid's own 'near surface water' rule can't
    tell sea from river/lake water, so a coastal sea cell reads as flooded just
    for being adjacent to itself -- the land mask is what actually removes that
    class of false positive."""
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))
    mosaic_path = modis_flood_mosaic_cache_path(str(tmp_path))
    band = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    lat, lon = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    save_jrc_hazard_mosaic(mosaic_path, band, lat, lon)
    land = np.array([[True, False], [False, True]])
    u._land_mask_cache.get.return_value = land

    with patch("atmos_gl.tasks.flood_risk.encode_frames") as mock_encode:
        u._render_live()

    encoded_band = mock_encode.call_args.args[0][0]
    assert encoded_band.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    call_lat, call_lon, call_shape = u._land_mask_cache.get.call_args.args
    assert call_lat.tolist() == lat.tolist()
    assert call_lon.tolist() == lon.tolist()
    assert call_shape == band.shape
    # dilate=False, exclude_lakes=True -- unlike every other LandMaskCache caller,
    # see _render_live's own docstring for why.
    assert u._land_mask_cache.get.call_args.kwargs == {
        "dilate": False,
        "exclude_lakes": True,
    }


def test_render_live_leaves_band_unmasked_when_land_mask_is_unavailable(tmp_path):
    """LandMaskCache.get returns None on geometry-load failure (no network for the
    one-time GSHHG fetch) -- render must still degrade gracefully, same contract
    every other LandMaskCache caller (currents/waves/FireWeatherUpdater) has."""
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))
    mosaic_path = modis_flood_mosaic_cache_path(str(tmp_path))
    band = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    save_jrc_hazard_mosaic(mosaic_path, band, np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    u._land_mask_cache.get.return_value = None

    with patch("atmos_gl.tasks.flood_risk.encode_frames") as mock_encode:
        u._render_live()

    encoded_band = mock_encode.call_args.args[0][0]
    assert encoded_band.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_render_historical_writes_a_texture_when_mosaic_is_cached(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))
    mosaic_path = jrc_hazard_mosaic_cache_path(str(tmp_path))
    band = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    save_jrc_hazard_mosaic(mosaic_path, band, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    out = u._render_historical()

    assert out == str(tmp_path / "data" / "flood_risk_historical.png")
    assert os.path.exists(out)
    assert os.path.exists(out + ".sig")
    assert _HISTORICAL_ENCODE_DOMAIN == (0.0, 4.0)


def test_render_historical_returns_none_when_mosaic_not_cached(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))

    assert u._render_historical() is None


def test_render_historical_skips_re_render_when_already_fresh(tmp_path):
    out_path = tmp_path / "data" / "flood_risk.png"
    u = _bare_updater_for_rendering(str(tmp_path), str(out_path))
    mosaic_path = jrc_hazard_mosaic_cache_path(str(tmp_path))
    band = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    save_jrc_hazard_mosaic(mosaic_path, band, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    first = u._render_historical()
    first_mtime = os.path.getmtime(first)

    second = u._render_historical()

    assert second == first
    assert os.path.getmtime(second) == first_mtime
