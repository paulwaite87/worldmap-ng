#!/usr/bin/env python3
"""Tests for the spec-driven scalar-field renderer (architecture review candidate "One
scalar-field renderer, many specs"). temperature/ozone/stormwatch had byte-identical
plot()/run() bodies differing only in a small spec; they now share ScalarFieldUpdater,
parametrised by ScalarFieldSpec. These tests lock the spec->render wiring so a fourth
field (one SPECS entry) can't silently mis-map cmap/range/extend/ticks/title, and guard
the partial() binding in layer_builder.
"""
from functools import partial
from unittest.mock import patch, MagicMock

import pytest

from atmos_gl.tasks.common import ForecastState
from atmos_gl.tasks.scalar_field import (
    ScalarFieldUpdater,
    ScalarFieldSpec,
    SPECS,
    _threshold_colormap,
)


def make_bare_updater(spec):
    """Bypass Updater.__init__ (does config/IO) and wire only what plot() reads."""
    u = ScalarFieldUpdater.__new__(ScalarFieldUpdater)
    u.spec = spec
    u.section = spec.product
    u.status_product = spec.product
    u.settings = {}
    u.common = {}
    u.map_region_bbox = (-180, -90, 180, 90)
    u.output_path = "/tmp/out/layer.png"
    u.map_data = MagicMock()
    u.map_data.region.region_identifier = "global"
    # Methods that touch IO / heavy libs are exercised elsewhere; stub them here.
    u.regrid_for_lod = MagicMock(return_value=([0], [0], [[0]]))
    u.get_output_path_for_hour = MagicMock(return_value="/tmp/out/layer_f003.png")
    return u


@pytest.mark.parametrize("key", ["temperature", "ozone", "stormwatch", "pwat"])
def test_plot_dispatches_spec_to_render(key):
    spec = SPECS[key]
    u = make_bare_updater(spec)
    field0 = {"lat": [0], "lon": [0], "values": [[1.0]]}
    state = ForecastState.at_hour("2026-06-13", "18", 3)

    with patch("atmos_gl.tasks.scalar_field.Plot") as MockPlot, patch(
        "atmos_gl.tasks.scalar_field.encode_frames"
    ) as mock_encode:
        u.plot(field0, state)

    # contourf gets this spec's extend mode (and the shared levels/zorder).
    contourf = MockPlot.return_value.ax.contourf
    contourf.assert_called_once()
    assert contourf.call_args.kwargs["extend"] == spec.extend
    assert contourf.call_args.kwargs["levels"] == 20
    assert contourf.call_args.kwargs["zorder"] == 2

    # The data texture is scaled to this spec's value range.
    assert mock_encode.call_args.args[2] == spec.vmin
    assert mock_encode.call_args.args[3] == spec.vmax


def test_plot_still_encodes_texture_when_contourf_raises():
    """The static PNG (contourf) and the GPU data texture (encode_frames) are
    independent outputs -- a contourf failure must not block the texture, since
    that's what the frontend's animated WebGL layer actually reads. Regression for a
    real, deterministic Cartopy bug (antimeridian-wrapping polygon reprojection
    inside contourf's own get_datalim() call) that reproduced for a specific ozone
    field on every attempt -- before this fix, that also permanently starved the
    hour's GPU texture, not just its (legacy) static render."""
    u = make_bare_updater(SPECS["ozone"])
    field0 = {"lat": [0], "lon": [0], "values": [[1.0]]}
    state = ForecastState.at_hour("2026-06-13", "18", 3)

    with patch("atmos_gl.tasks.scalar_field.Plot") as MockPlot, patch(
        "atmos_gl.tasks.scalar_field.encode_frames"
    ) as mock_encode:
        MockPlot.return_value.ax.contourf.side_effect = ValueError(
            "Sequences of multi-polygons are not valid arguments"
        )
        u.plot(field0, state)  # must not raise

    mock_encode.assert_called_once()


def test_plot_skips_when_field_missing():
    u = make_bare_updater(SPECS["temperature"])
    state = ForecastState.at_hour("2026-06-13", "18", 3)
    with patch("atmos_gl.tasks.scalar_field.Plot") as MockPlot, patch(
        "atmos_gl.tasks.scalar_field.encode_frames"
    ):
        u.plot({"lat": [0], "lon": [0], "values": None}, state)
    MockPlot.assert_not_called()


def test_specs_cover_the_four_scalar_fields():
    assert set(SPECS) == {"temperature", "ozone", "stormwatch", "pwat"}
    for key, spec in SPECS.items():
        assert isinstance(spec, ScalarFieldSpec)
        assert spec.product == key


def test_layer_builder_binds_each_spec_via_partial():
    from atmos_gl.layer_builder import TASK_CLASSES

    for key in SPECS:
        entry = TASK_CLASSES[key]
        assert isinstance(entry, partial)
        assert entry.func is ScalarFieldUpdater
        assert entry.keywords["spec"] is SPECS[key]


class TestThresholdColormap:
    """_threshold_colormap: the ozone/pwat 'critical zone' rendering (architecture
    review candidate #5 -- restores ozone's dropped critical-palette behaviour and
    reuses the same mechanism for pwat, mirrored)."""

    def test_focus_below_grades_toward_vmin_flat_above_threshold(self):
        # ozone shape: worst reading is at vmin: palette[-1] (yellow) should land there;
        # palette[0] (magenta) anchors the threshold boundary; above threshold is flat.
        magenta, yellow = (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)
        flat = (0.0, 0.1, 0.3, 0.2)
        cmap = _threshold_colormap(
            vmin=150.0, vmax=500.0, threshold=220.0, focus="below",
            palette_colors=[magenta, yellow], flat_color=flat,
        )
        t = (220.0 - 150.0) / (500.0 - 150.0)
        assert cmap(0.0)[:3] == pytest.approx(yellow, abs=0.02)
        assert cmap(t)[:3] == pytest.approx(magenta, abs=0.02)
        assert cmap(1.0) == pytest.approx(flat, abs=0.02)
        assert cmap(0.9) == pytest.approx(flat, abs=0.02)

    def test_focus_above_grades_toward_vmax_flat_below_threshold(self):
        # pwat shape: worst reading is at vmax: palette[-1] anchors vmax, palette[0]
        # anchors the threshold boundary; below threshold is flat (invisible).
        blue, violet = (0.0, 0.0, 0.55), (0.6, 0.0, 0.85)
        flat = (0.0, 0.0, 0.0, 0.0)
        cmap = _threshold_colormap(
            vmin=0.0, vmax=80.0, threshold=50.0, focus="above",
            palette_colors=[blue, violet], flat_color=flat,
        )
        t = 50.0 / 80.0
        assert cmap(t)[:3] == pytest.approx(blue, abs=0.02)
        assert cmap(1.0)[:3] == pytest.approx(violet, abs=0.02)
        assert cmap(0.0) == pytest.approx(flat, abs=0.02)
        assert cmap(t - 0.1) == pytest.approx(flat, abs=0.02)

    def test_multi_stop_palette_spans_boundary_to_extreme(self):
        # pwat's "standard" palette (precipitation's 7-stop ramp): first colour at the
        # threshold boundary, last colour at vmax.
        cyan = (0.0, 1.0, 1.0)
        magenta = (1.0, 0.0, 1.0)
        palette = [cyan, (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), magenta]
        cmap = _threshold_colormap(
            vmin=0.0, vmax=80.0, threshold=50.0, focus="above",
            palette_colors=palette, flat_color=(0, 0, 0, 0),
        )
        t = 50.0 / 80.0
        assert cmap(t)[:3] == pytest.approx(cyan, abs=0.02)
        assert cmap(1.0)[:3] == pytest.approx(magenta, abs=0.02)


def test_resolve_cmap_reads_live_threshold_and_palette_settings():
    u = make_bare_updater(SPECS["pwat"])
    u.settings = {"critical_pwat": 30.0, "palette": "deep_teal"}
    cmap = u._resolve_cmap()
    pale_cyan = (0.7, 1.0, 1.0)
    deep_teal = (0.0, 0.35, 0.3)
    t = 30.0 / 80.0
    assert cmap(t)[:3] == pytest.approx(pale_cyan, abs=0.02)
    assert cmap(1.0)[:3] == pytest.approx(deep_teal, abs=0.02)


def test_resolve_cmap_falls_back_to_spec_defaults_when_unset():
    u = make_bare_updater(SPECS["ozone"])
    u.settings = {}
    cmap = u._resolve_cmap()
    yellow = (1.0, 1.0, 0.0)
    assert cmap(0.0)[:3] == pytest.approx(yellow, abs=0.02)


def test_resolve_cmap_plain_specs_unaffected():
    u = make_bare_updater(SPECS["temperature"])
    cmap = u._resolve_cmap()
    assert cmap.name == "RdYlBu_r"


@pytest.mark.parametrize("key", ["temperature", "ozone", "stormwatch", "pwat"])
def test_mask_values_defaults_to_a_noop_passthrough(key):
    """Only FireWeatherUpdater overrides this (issue #390) -- every plain SPECS
    entry is meaningful everywhere on the globe and must never have its values
    touched by the hook."""
    u = make_bare_updater(SPECS[key])
    values = [[1.0, 2.0], [3.0, 4.0]]
    assert u._mask_values(values, [0, 1], [0, 1]) is values
