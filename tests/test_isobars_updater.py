#!/usr/bin/env python3
"""Tests for IsobarUpdater's render-settings signature (bug: changing linewidth in
the config UI never redrew already-cached hours -- see should_plot_for_hour's
settings_sig parameter in test_multi_hour_render_mixin.py /
test_common_render_all_hours.py for the shared mechanism this wires into).

should_plot_for_hour only compared the output file's mtime against the DATA's
updated_at, so a settings-only edit (linewidth/isobar_color/isobar_step/
label_fontsize/opacity -- all baked directly into the PNG by IsobarUpdater.plot())
touched neither and was silently never picked up. These tests lock
_render_settings_signature's reaction to each of those settings, and that run()
actually forwards it to render_all_hours.
"""
from unittest.mock import MagicMock

from atmos_gl.tasks.isobars import IsobarUpdater


def make_bare_updater(settings=None):
    u = IsobarUpdater.__new__(IsobarUpdater)
    u.settings = settings or {}
    return u


def test_render_settings_signature_stable_for_identical_settings():
    u = make_bare_updater({"linewidth": 1.5, "isobar_color": "white"})
    assert u._render_settings_signature() == u._render_settings_signature()


def test_render_settings_signature_changes_when_linewidth_changes():
    u1 = make_bare_updater({"linewidth": 1.0})
    u2 = make_bare_updater({"linewidth": 2.0})
    assert u1._render_settings_signature() != u2._render_settings_signature()


def test_render_settings_signature_changes_for_each_render_relevant_setting():
    base = {
        "isobar_step": 4,
        "isobar_color": "white",
        "linewidth": 1.0,
        "label_fontsize": 10,
        "opacity": 100,
    }
    base_sig = make_bare_updater(dict(base))._render_settings_signature()
    for key, changed in (
        ("isobar_step", 8),
        ("isobar_color", "black"),
        ("linewidth", 3.0),
        ("label_fontsize", 14),
        ("opacity", 60),
    ):
        variant = dict(base)
        variant[key] = changed
        variant_sig = make_bare_updater(variant)._render_settings_signature()
        assert variant_sig != base_sig, f"{key} change did not alter the signature"


def test_run_forwards_render_settings_signature_to_render_all_hours():
    u = make_bare_updater({"linewidth": 2.0})
    u.get_gfs_state = MagicMock()
    u.render_all_hours = MagicMock(return_value=1)

    u.run(max_hours=1)

    u.render_all_hours.assert_called_once()
    kwargs = u.render_all_hours.call_args.kwargs
    assert kwargs["settings_sig"] == u._render_settings_signature()
    assert kwargs["max_hours"] == 1
