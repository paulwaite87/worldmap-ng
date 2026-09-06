#!/usr/bin/env python3
"""Tests for SingleHourScalarUpdater -- the shared control flow behind
ScalarFieldUpdater and PrecipitationUpdater (architecture review candidate "give
precipitation the ScalarFieldSpec deepening its frontend already got"). Both
subclasses' plot() bodies diverge too much to share (different regrid fill_value,
color models, and precipitation's extra smoothing/floor/sqrt-transform pipeline), so
only the genuinely-identical constructor fields and run() wiring were lifted here --
plot() stays a full per-subclass override, not hook-split. This locks run()'s call
order (get_gfs_state -> render_all_hours) and hook dispatch, since neither
ScalarFieldUpdater.run() nor PrecipitationUpdater.run() had direct coverage before
this extraction.
"""
from unittest.mock import MagicMock, patch

from atmos_gl.tasks.single_hour_scalar import SingleHourScalarUpdater


class _StubUpdater(SingleHourScalarUpdater):
    def __init__(self):
        # Bypass Updater.__init__ (config/IO) -- wire only what run() reads.
        self.settings = {"level_of_detail": 2}
        self.section = "stub"
        self.status_product = "stub"
        self.per_hour_outputs = [".png", "_data.png"]
        self.level_of_detail = 2
        self.lod_desc = None
        self.plot_calls = []

    def plot(self, field0, state):
        self.plot_calls.append((field0, state))


def make_stub():
    u = _StubUpdater()
    u.get_gfs_state = MagicMock()
    u.render_all_hours = MagicMock()
    return u


def test_run_calls_get_gfs_state_then_render_all_hours_in_order():
    u = make_stub()
    manager = MagicMock()
    manager.attach_mock(u.get_gfs_state, "get_gfs_state")
    manager.attach_mock(u.render_all_hours, "render_all_hours")

    u.run()

    assert [c[0] for c in manager.mock_calls] == [
        "get_gfs_state",
        "render_all_hours",
    ]


def test_run_dispatches_render_all_hours_with_status_product_plot_and_max_hours():
    u = make_stub()

    u.run(max_hours=1)

    u.render_all_hours.assert_called_once()
    args, kwargs = u.render_all_hours.call_args
    assert args[0] == "stub"
    assert kwargs["plot_fn"] == u.plot
    assert kwargs["max_hours"] == 1
    assert callable(kwargs["field_ready"])
    assert kwargs["field_ready"]({"values": [1]}) is True
    assert kwargs["field_ready"]({"values": None}) is False


def test_render_settings_signature_defaults_to_none():
    """The base class bakes nothing settings-derived into pixels itself (that's
    entirely per-subclass, e.g. PrecipitationUpdater's min_mm_hr/opacity/palette) --
    None keeps should_plot_for_hour's old data-only freshness check for any subclass
    that doesn't override this."""
    u = _StubUpdater()
    assert u._render_settings_signature() is None


def test_run_forwards_render_settings_signature_to_render_all_hours():
    u = make_stub()

    u.run(max_hours=1)

    kwargs = u.render_all_hours.call_args.kwargs
    assert kwargs["settings_sig"] is None


def test_run_returns_render_all_hours_result():
    u = make_stub()
    u.render_all_hours.return_value = "sentinel"

    assert u.run() == "sentinel"


def test_plot_raises_not_implemented_on_the_bare_base():
    bare = SingleHourScalarUpdater.__new__(SingleHourScalarUpdater)

    try:
        bare.plot(None, None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError from plot")


def test_constructor_sets_shared_fields_from_product():
    def fake_updater_init(self, config, section, map_data):
        self.section = section
        self.settings = {"level_of_detail": 3}

    with patch(
        "atmos_gl.tasks.single_hour_scalar.Updater.__init__", fake_updater_init
    ):
        u = SingleHourScalarUpdater(
            config=MagicMock(), product="precipitation", map_data=MagicMock()
        )

    assert u.level_of_detail == 3
    assert u.lod_desc is None
    assert u.per_hour_outputs == [".png", "_data.png"]
    assert u.status_product == "precipitation"
