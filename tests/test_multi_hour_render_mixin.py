#!/usr/bin/env python3
"""Tests for the Updater / MultiHourRenderMixin split (architecture review candidate
"slim Updater"). get_output_path_for_hour/publish_current_hour/should_plot_for_hour/
render_all_hours moved off Updater itself onto a mixin that only multi-hour layers
(isobars, wind, precipitation, currents, waves, the scalar-field trio) inherit --
single-shot layers (sst, clouds, markers) no longer see them at all. These 4 methods'
own logic is unchanged (a verbatim move); these tests lock the architectural split
itself and smoke-test the mixin still works when mixed into a bare instance.
"""
import os
from datetime import datetime, timedelta, timezone

from atmos_gl.tasks.common import Updater, MultiHourRenderMixin, ForecastState


def test_updater_itself_does_not_have_the_per_hour_methods():
    for name in (
        "render_all_hours",
        "should_plot_for_hour",
        "publish_current_hour",
        "get_output_path_for_hour",
    ):
        assert not hasattr(Updater, name), f"Updater should not define {name}"


def test_updater_still_has_get_db_field_at_hour():
    """get_db_field_at_hour stays on Updater itself (not the mixin) -- markers.py, a
    single-shot layer, calls it directly to sample weather at a specific hour, not to
    render a per-hour output."""
    assert hasattr(Updater, "get_db_field_at_hour")


def test_mixin_exposes_exactly_the_four_per_hour_methods():
    own_methods = {
        name
        for name in vars(MultiHourRenderMixin)
        if not name.startswith("__") and callable(getattr(MultiHourRenderMixin, name))
    }
    assert own_methods == {
        "render_all_hours",
        "should_plot_for_hour",
        "publish_current_hour",
        "get_output_path_for_hour",
    }


class _MultiHourLayer(Updater, MultiHourRenderMixin):
    pass


def make_bare_multi_hour_layer(output_path, per_hour_outputs=None):
    u = _MultiHourLayer.__new__(_MultiHourLayer)
    u.section = "test"
    u.output_path = output_path
    u.per_hour_outputs = per_hour_outputs or [".png"]
    return u


def test_get_output_path_for_hour_requires_an_explicit_hour(tmp_path):
    """fhour has no self-fallback (architecture review candidate "ForecastState full
    thread-through") -- every caller passes it explicitly now."""
    u = make_bare_multi_hour_layer(str(tmp_path / "isobars.png"))
    try:
        u.get_output_path_for_hour()
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError: fhour is a required argument")


def test_get_output_path_for_hour_accepts_explicit_hour(tmp_path):
    u = make_bare_multi_hour_layer(str(tmp_path / "isobars.png"))
    assert u.get_output_path_for_hour(12) == str(tmp_path / "isobars_f012.png")


def test_should_plot_for_hour_true_when_output_missing(tmp_path):
    u = make_bare_multi_hour_layer(str(tmp_path / "isobars.png"))
    state = ForecastState.at_hour("2026-06-13", "18", 3)
    assert u.should_plot_for_hour(state, "isobars") is True


class _FakeStore:
    """Reports the data as already written `age` ago -- older than the (freshly
    touched) output file, so should_plot_for_hour's data-freshness check reads as
    'fresh', isolating the settings_sig check the tests below exercise."""

    def __init__(self, age=timedelta(hours=1)):
        self._updated_at = datetime.now(timezone.utc) - age

    def get_field_meta(self, run_date_str, run_id, fhour, product_name):
        return {"updated_at": self._updated_at}


def make_complete_multi_hour_layer(tmp_path):
    """A bare multi-hour layer whose single required output (isobars_f003.png) exists
    and is data-fresh -- isolates should_plot_for_hour's settings_sig branch from the
    unrelated missing-file / data-mtime checks."""
    base = str(tmp_path / "isobars.png")
    u = make_bare_multi_hour_layer(base)
    u._store = _FakeStore()
    out = tmp_path / "isobars_f003.png"
    out.write_bytes(b"fake-png-bytes")
    return u, str(out)


def test_should_plot_for_hour_true_when_settings_sig_sidecar_missing(tmp_path):
    """A settings_sig is given but no '<out>.sig' was ever written (e.g. rendered
    before this check existed, or under a settings-blind caller) -- treated as stale,
    not silently trusted."""
    u, _out = make_complete_multi_hour_layer(tmp_path)
    state = ForecastState.at_hour("2026-06-13", "18", 3)
    assert u.should_plot_for_hour(state, "isobars", settings_sig="sig-a") is True


def test_should_plot_for_hour_true_when_settings_sig_mismatches(tmp_path):
    """The actual bug this guards against: data unchanged, but the config-derived
    signature (e.g. isobars' linewidth) differs from what was last rendered."""
    u, out = make_complete_multi_hour_layer(tmp_path)
    u._write_render_signature(out, "old-sig")
    state = ForecastState.at_hour("2026-06-13", "18", 3)
    assert u.should_plot_for_hour(state, "isobars", settings_sig="new-sig") is True


def test_should_plot_for_hour_false_when_settings_sig_matches_and_data_fresh(tmp_path):
    u, out = make_complete_multi_hour_layer(tmp_path)
    u._write_render_signature(out, "sig-a")
    state = ForecastState.at_hour("2026-06-13", "18", 3)
    assert u.should_plot_for_hour(state, "isobars", settings_sig="sig-a") is False


def test_should_plot_for_hour_ignores_settings_sig_when_not_given(tmp_path):
    """Callers that don't pass settings_sig (the default, None) keep the old
    data-only freshness behaviour -- no '.sig' sidecar required."""
    u, _out = make_complete_multi_hour_layer(tmp_path)
    state = ForecastState.at_hour("2026-06-13", "18", 3)
    assert u.should_plot_for_hour(state, "isobars") is False


def test_publish_current_hour_copies_per_hour_output_to_base_name(tmp_path):
    base = str(tmp_path / "isobars.png")
    u = make_bare_multi_hour_layer(base)
    per_hour_path = tmp_path / "isobars_f003.png"
    per_hour_path.write_bytes(b"fake-png-bytes")

    u.publish_current_hour(3)

    assert os.path.exists(base)
    assert (tmp_path / "isobars.png").read_bytes() == b"fake-png-bytes"
