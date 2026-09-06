#!/usr/bin/env python3
"""Tests for FireWeatherUpdater (tasks/fire_weather.py). Unlike every ScalarFieldSpec
entry in tests/test_scalar_field.py, this task deliberately decouples its config
section ("fires", shared with the FIRMS collector/route) from its fieldstore product
("fire_weather") -- these tests exist specifically to pin that decoupling, since
ScalarFieldUpdater.__init__ normally forces them to be the same string.
"""
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from atmos_gl.tasks.common import ForecastState
from atmos_gl.tasks.fire_weather import FireWeatherUpdater, FIRE_WEATHER_SPEC
from atmos_gl.tasks.scalar_field import ScalarFieldUpdater


def make_bare_fire_weather_updater():
    """Bypass Updater.__init__ (does config/IO), mirroring
    test_scalar_field.py's make_bare_updater. Land/vegetation caches default to
    "no mask data available" (both .get() calls return None), so plot()-level
    tests exercise pure spec-dispatch behaviour unaffected by masking -- masking
    itself is covered separately below."""
    u = FireWeatherUpdater.__new__(FireWeatherUpdater)
    u.spec = FIRE_WEATHER_SPEC
    u.section = "fires"
    u.status_product = "fire_weather"
    u.settings = {}
    u.map_region_bbox = (-180, -90, 180, 90)
    u.output_path = "/tmp/out/fires.png"
    u.map_data = MagicMock()
    u.map_data.region.region_identifier = "global"
    u.regrid_for_lod = MagicMock(return_value=([0], [0], [[0]]))
    u.get_output_path_for_hour = MagicMock(return_value="/tmp/out/fires_f003.png")
    u._land_mask_cache = MagicMock(get=MagicMock(return_value=None))
    u._vegetation_mask_cache = MagicMock(get=MagicMock(return_value=None))
    return u


def test_init_decouples_section_from_fieldstore_product():
    """The whole reason this task exists instead of a plain SPECS entry: section
    ("fires", shared with the DB collector's config) and status_product/fieldstore key
    ("fire_weather", distinct) must differ."""
    config = MagicMock()
    config.get_section.return_value = {"level_of_detail": "2"}
    map_data = MagicMock()

    with patch("atmos_gl.tasks.common.fieldstore.make_store"), patch(
        "atmos_gl.tasks.common.ProcessStatusAdapter"
    ):
        u = FireWeatherUpdater(config, map_data)

    assert u.section == "fires"
    assert u.status_product == "fire_weather"
    assert u.spec.product == "fire_weather"
    assert u.spec is FIRE_WEATHER_SPEC
    assert u.level_of_detail == 2
    assert u.per_hour_outputs == [".png", "_data.png"]


def test_is_a_scalar_field_updater_and_reuses_its_render_path():
    """plot()/_resolve_cmap()/run() must all be the inherited ScalarFieldUpdater
    implementations, unchanged -- this task overrides ONLY __init__."""
    assert issubclass(FireWeatherUpdater, ScalarFieldUpdater)
    assert FireWeatherUpdater.plot is ScalarFieldUpdater.plot
    assert FireWeatherUpdater.run is ScalarFieldUpdater.run
    assert FireWeatherUpdater._resolve_cmap is ScalarFieldUpdater._resolve_cmap


def test_plot_dispatches_the_fire_weather_spec():
    u = make_bare_fire_weather_updater()
    field0 = {"lat": [0], "lon": [0], "values": [[42.0]]}
    state = ForecastState.at_hour("2026-06-13", "18", 3)

    with patch("atmos_gl.tasks.scalar_field.Plot") as MockPlot, patch(
        "atmos_gl.tasks.scalar_field.encode_frames"
    ) as mock_encode:
        u.plot(field0, state)

    contourf = MockPlot.return_value.ax.contourf
    contourf.assert_called_once()
    assert contourf.call_args.kwargs["extend"] == "max"
    assert mock_encode.call_args.args[2] == 0.0
    assert mock_encode.call_args.args[3] == 100.0


def test_resolve_cmap_uses_the_single_fixed_palette_by_default():
    """No ("fires", "palette") setting is exposed in config -- must fall back to the
    spec's own palette_default regardless of what's in settings."""
    u = make_bare_fire_weather_updater()
    u.settings = {}
    cmap = u._resolve_cmap()
    pale_yellow = (1.0, 0.95, 0.6)
    deep_red = (0.6, 0.0, 0.0)

    # Default threshold (min_risk_display) is 25/100 -- the palette's first colour
    # anchors that boundary; below it renders flat/transparent (see the threshold test).
    t = 25.0 / 100.0
    assert cmap(t)[:3] == pytest.approx(pale_yellow, abs=0.02)
    assert cmap(1.0)[:3] == pytest.approx(deep_red, abs=0.02)


def test_resolve_cmap_reads_live_min_risk_display_threshold():
    u = make_bare_fire_weather_updater()
    u.settings = {"min_risk_display": 50.0}
    cmap = u._resolve_cmap()
    pale_yellow = (1.0, 0.95, 0.6)

    t = 50.0 / 100.0
    assert cmap(t)[:3] == pytest.approx(pale_yellow, abs=0.02)
    assert cmap(t - 0.1) == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=0.02)


def test_layer_builder_registers_fires_as_fire_weather_updater():
    from atmos_gl.layer_builder import TASK_CLASSES

    assert TASK_CLASSES["fires"] is FireWeatherUpdater


# ---- Land + vegetation masking (issue #390) ---------------------------------


def test_init_wires_land_and_vegetation_mask_caches():
    config = MagicMock()
    config.get_section.return_value = {}
    map_data = MagicMock()

    with patch("atmos_gl.tasks.common.fieldstore.make_store"), patch(
        "atmos_gl.tasks.common.ProcessStatusAdapter"
    ), patch("atmos_gl.tasks.fire_weather.LandMaskCache") as MockLandCache, patch(
        "atmos_gl.tasks.fire_weather.VegetationMaskCache"
    ) as MockVegCache:
        u = FireWeatherUpdater(config, map_data)

    MockLandCache.assert_called_once_with("FireWeather")
    MockVegCache.assert_called_once_with("FireWeather", u.workdir)
    assert u._land_mask_cache is MockLandCache.return_value
    assert u._vegetation_mask_cache is MockVegCache.return_value


def test_mask_values_ands_land_and_vegetation_together():
    u = make_bare_fire_weather_updater()
    # 2x2 grid: land only top row, burnable only left column -> only top-left
    # cell passes both.
    u._land_mask_cache.get.return_value = np.array([[True, True], [False, False]])
    u._vegetation_mask_cache.get.return_value = np.array([[True, False], [True, False]])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = u._mask_values(values, [10.0, 0.0], [0.0, 10.0])

    assert result[0, 0] == 1.0
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 0])
    assert np.isnan(result[1, 1])


def test_mask_values_falls_back_to_land_only_when_vegetation_mask_unavailable():
    u = make_bare_fire_weather_updater()
    u._land_mask_cache.get.return_value = np.array([[True, False]])
    u._vegetation_mask_cache.get.return_value = None
    values = np.array([[1.0, 2.0]])

    result = u._mask_values(values, [0.0], [0.0, 10.0])

    assert result[0, 0] == 1.0
    assert np.isnan(result[0, 1])


def test_mask_values_falls_back_to_vegetation_only_when_land_mask_unavailable():
    u = make_bare_fire_weather_updater()
    u._land_mask_cache.get.return_value = None
    u._vegetation_mask_cache.get.return_value = np.array([[False, True]])
    values = np.array([[1.0, 2.0]])

    result = u._mask_values(values, [0.0], [0.0, 10.0])

    assert np.isnan(result[0, 0])
    assert result[0, 1] == 2.0


def test_mask_values_returns_values_unchanged_when_both_masks_unavailable():
    """Today's fully-unmasked behaviour (the original bug) is the correct,
    deliberate fallback when neither mask has any data yet -- see issue #390's
    "Fallback / degradation behavior" decision."""
    u = make_bare_fire_weather_updater()
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = u._mask_values(values, [10.0, 0.0], [0.0, 10.0])

    assert (result == values).all()


def test_mask_values_uses_a_cache_key_that_distinguishes_native_from_lod_grids():
    """The native (descending-lat) and LOD-regridded (ascending-lat) grids can
    coincidentally share a shape, but must never share a cache key -- otherwise
    one grid's mask would get silently, and wrongly, reused for the other (wrong
    content AND wrong latitude ordering)."""
    u = make_bare_fire_weather_updater()
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    u._mask_values(values, [10.0, 0.0], [0.0, 10.0])  # native: descending
    u._mask_values(values, [0.0, 10.0], [0.0, 10.0])  # LOD: ascending, same shape

    land_keys = [call.args[2] for call in u._land_mask_cache.get.call_args_list]
    assert len(land_keys) == 2
    assert land_keys[0] != land_keys[1]
