#!/usr/bin/env python3
"""Tests for MultiHourRenderMixin.render_all_hours (architecture review candidate
"ForecastState full thread-through"). This method just underwent its biggest rewrite
of that candidate -- no more self.run_date_str/run_id/forecast_hour_str mutation or
try/finally restore, a fresh ForecastState built per hour, and plot_fn's signature
changed to plot_fn(field, state) -- and had zero direct test coverage before this
(only exercised indirectly through subclass tests that mock plot() itself).
"""
from unittest.mock import MagicMock

from atmos_gl.tasks.common import Updater, MultiHourRenderMixin, ForecastState


class _MultiHourLayer(Updater, MultiHourRenderMixin):
    pass


def make_bare_layer(hours_resolved=("2026-06-13", "18", [0, 3, 6])):
    u = _MultiHourLayer.__new__(_MultiHourLayer)
    u.section = "test"
    u.output_path = "/tmp/out/test.png"
    u.per_hour_outputs = [".png"]
    u.latest_store_run = MagicMock(return_value=hours_resolved)
    u.process_status_adapter = MagicMock()
    u.publish_current_hour = MagicMock()
    # Default: nothing is "before now", matching every pre-existing test's expectation
    # that all resolved hours are considered. test_skips_hours_before_now_fhour(_*)
    # override this directly to exercise the filter itself.
    u._now_fhour = MagicMock(return_value=0)
    return u


def _pending_then_done(already_done=()):
    """should_plot_for_hour stub paired with a plot_fn side_effect: an hour reads as
    "needs plotting" (True) until mark_done() has been called for it -- mirrors real
    should_plot_for_hour, which flips False once plot_fn has actually written the
    hour's outputs. render_all_hours re-checks should_plot_for_hour right after a
    non-raising plot_fn call to decide whether the hour counts toward max_hours (a
    plot_fn that swallows its own partial failure, like WindUpdater/ScalarFieldUpdater,
    can return without raising yet leave the hour incomplete -- see render_all_hours'
    docstring) -- so tests simulating "this hour rendered successfully" must flip the
    same way, not just return a constant.
    """
    done = set(already_done)

    def should_plot(state, product):
        return state.fhour not in done

    def mark_done(field, state):
        done.add(state.fhour)

    return should_plot, mark_done


def test_builds_a_fresh_state_per_hour_and_passes_it_to_plot_fn():
    u = make_bare_layer()
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    plot_fn = MagicMock(side_effect=mark_done)

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    assert plotted == 3
    seen_hours = [call.args[1].fhour for call in plot_fn.call_args_list]
    assert seen_hours == [0, 3, 6]
    for call in plot_fn.call_args_list:
        state = call.args[1]
        assert isinstance(state, ForecastState)
        assert state.run_date_str == "2026-06-13"
        assert state.run_id == "18"


def test_skips_hours_that_are_already_fresh():
    u = make_bare_layer()
    should_plot, mark_done = _pending_then_done(already_done=(3,))
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    plot_fn = MagicMock(side_effect=mark_done)

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    assert plotted == 2
    seen_hours = {call.args[1].fhour for call in plot_fn.call_args_list}
    assert seen_hours == {0, 6}


def test_skips_hours_whose_field_is_not_ready():
    u = make_bare_layer()
    u.should_plot_for_hour = MagicMock(return_value=True)
    u.get_db_field_at_hour = MagicMock(return_value={"u": None})
    plot_fn = MagicMock()

    plotted = u.render_all_hours("wind", plot_fn, field_ready=lambda f: f.get("u") is not None)

    assert plotted == 0
    plot_fn.assert_not_called()


def test_one_hour_plot_failure_does_not_stop_the_others():
    u = make_bare_layer()
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})

    def flaky_plot(field, state):
        if state.fhour == 3:
            raise RuntimeError("boom")
        mark_done(field, state)

    plotted = u.render_all_hours("isobars", flaky_plot, field_ready=lambda f: True)

    assert plotted == 2  # hours 0 and 6 still rendered despite hour 3 raising


def test_a_hour_that_never_completes_does_not_block_later_hours():
    """Regression: a plot_fn that swallows its own partial failure (WindUpdater/
    ScalarFieldUpdater's contourf-vs-texture split, issue #283) never raises, so
    "plot_fn didn't raise" alone can't be trusted as "this hour is done". Before
    render_all_hours re-checked should_plot_for_hour, a permanently-broken earliest
    hour would count as plotted every call, get re-selected first again next call
    (still earliest, still pending), and -- under max_hours=1's round-robin dispatch --
    starve every later hour behind it forever (observed live: Data Status stuck at 0%,
    logs repeating "static render fNNN failed" for one hour)."""
    u = make_bare_layer()
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})

    def _plot(field, state):
        if state.fhour != 0:
            mark_done(field, state)  # hour 0 never actually completes

    plot_fn = MagicMock(side_effect=_plot)

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True, max_hours=1)

    assert plotted == 1
    seen_hours = [call.args[1].fhour for call in plot_fn.call_args_list]
    # hour 0 is attempted (and published) but not credited; the loop moves on and
    # yields as soon as hour 3 genuinely completes, instead of stopping cold on 0.
    assert seen_hours == [0, 3]
    assert u.publish_current_hour.call_args_list == [((0,),), ((3,),)]


def test_publishes_each_plotted_hour_immediately():
    """Publishes as each hour lands, not once at the end -- with max_hours capping a
    call to one hour, "once at the end" would mean "never" until the whole backlog
    drains (architecture review candidate "interleave per-hour rendering across
    layers")."""
    u = make_bare_layer()
    u.should_plot_for_hour = MagicMock(return_value=True)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    plot_fn = MagicMock()

    u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    assert u.publish_current_hour.call_args_list == [
        ((0,),), ((3,),), ((6,),),
    ]


def test_does_not_publish_when_nothing_needs_plotting():
    u = make_bare_layer()
    u.should_plot_for_hour = MagicMock(return_value=False)  # nothing needs plotting
    u.get_db_field_at_hour = MagicMock()
    plot_fn = MagicMock()

    u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    u.publish_current_hour.assert_not_called()


def test_max_hours_stops_after_plotting_that_many_and_yields():
    u = make_bare_layer()
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    plot_fn = MagicMock(side_effect=mark_done)

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True, max_hours=1)

    assert plotted == 1
    seen_hours = [call.args[1].fhour for call in plot_fn.call_args_list]
    assert seen_hours == [0]  # stopped after the first hour, never examined 3 or 6
    u.publish_current_hour.assert_called_once_with(0)


def test_max_hours_none_drains_the_whole_backlog():
    u = make_bare_layer()
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    plot_fn = MagicMock(side_effect=mark_done)

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True, max_hours=None)

    assert plotted == 3


def test_no_catalog_data_returns_zero_and_does_not_publish():
    u = make_bare_layer(hours_resolved=None)
    plot_fn = MagicMock()

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    assert plotted == 0
    plot_fn.assert_not_called()
    u.publish_current_hour.assert_not_called()


def test_skips_hours_before_now_fhour():
    """Hours already in the catalog but before "now" are never reachable via the
    scrubber (layer_status()'s own segments already filter them out -- see
    Updater.layer_status()'s docstring), so rendering them wastes a max_hours=1
    round-robin turn on output nobody can ever see. A section re-enabled after being
    off for a while (e.g. a channel toggle) should catch up starting from "now", not
    replay its entire past backlog first."""
    u = make_bare_layer(hours_resolved=("2026-06-13", "18", [0, 3, 6, 9]))
    u._now_fhour = MagicMock(return_value=6)
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    plot_fn = MagicMock(side_effect=mark_done)

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    assert plotted == 2
    seen_hours = [call.args[1].fhour for call in plot_fn.call_args_list]
    assert seen_hours == [6, 9]  # 0 and 3 are before "now" -- skipped entirely


def test_skips_hours_before_now_fhour_respects_max_hours():
    u = make_bare_layer(hours_resolved=("2026-06-13", "18", [0, 3, 6, 9]))
    u._now_fhour = MagicMock(return_value=6)
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    plot_fn = MagicMock(side_effect=mark_done)

    plotted = u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True, max_hours=1)

    assert plotted == 1
    seen_hours = [call.args[1].fhour for call in plot_fn.call_args_list]
    assert seen_hours == [6]  # the first reachable hour, not 0


def test_settings_sig_is_forwarded_to_should_plot_for_hour_when_given():
    """render_all_hours must thread settings_sig through to every should_plot_for_hour
    call -- both the initial per-hour check and the post-plot recheck -- otherwise a
    settings-only change (the isobars linewidth bug) is invisible to the mixin."""
    u = make_bare_layer(hours_resolved=("2026-06-13", "18", [3]))
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(
        side_effect=lambda state, product, **kw: should_plot(state, product)
    )
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    u._write_render_signature = MagicMock()
    plot_fn = MagicMock(side_effect=mark_done)

    u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True, settings_sig="sig-a")

    assert u.should_plot_for_hour.call_args_list
    for call in u.should_plot_for_hour.call_args_list:
        assert call.kwargs.get("settings_sig") == "sig-a"


def test_no_settings_sig_kwarg_when_none_given():
    """Callers that don't pass settings_sig (every pre-existing caller) must keep
    calling should_plot_for_hour with its original two positional args -- no
    settings_sig kwarg at all -- so mocks/subclasses written against that shape are
    unaffected."""
    u = make_bare_layer()
    u.should_plot_for_hour = MagicMock(return_value=False)
    plot_fn = MagicMock()

    u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    for call in u.should_plot_for_hour.call_args_list:
        assert call.kwargs == {}


def test_writes_settings_signature_sidecar_after_successful_plot():
    u = make_bare_layer(hours_resolved=("2026-06-13", "18", [3]))
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(
        side_effect=lambda state, product, **kw: should_plot(state, product)
    )
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    u._write_render_signature = MagicMock()
    plot_fn = MagicMock(side_effect=mark_done)

    u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True, settings_sig="sig-a")

    u._write_render_signature.assert_called_once_with(u.get_output_path_for_hour(3), "sig-a")


def test_does_not_write_signature_sidecar_when_no_settings_sig_given():
    u = make_bare_layer(hours_resolved=("2026-06-13", "18", [3]))
    should_plot, mark_done = _pending_then_done()
    u.should_plot_for_hour = MagicMock(side_effect=should_plot)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})
    u._write_render_signature = MagicMock()
    plot_fn = MagicMock(side_effect=mark_done)

    u.render_all_hours("isobars", plot_fn, field_ready=lambda f: True)

    u._write_render_signature.assert_not_called()


def test_no_instance_state_is_mutated():
    """The whole point of full thread-through: render_all_hours must not read or write
    self.run_date_str/run_id/forecast_hour_str at all."""
    u = make_bare_layer()
    u.should_plot_for_hour = MagicMock(return_value=True)
    u.get_db_field_at_hour = MagicMock(return_value={"values": [[1.0]]})

    u.render_all_hours("isobars", MagicMock(), field_ready=lambda f: True)

    assert not hasattr(u, "run_date_str")
    assert not hasattr(u, "run_id")
    assert not hasattr(u, "forecast_hour_str")
