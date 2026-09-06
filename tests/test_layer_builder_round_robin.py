#!/usr/bin/env python3
"""Tests for layer_builder.py's round-robin per-hour dispatch (architecture review
candidate "interleave per-hour rendering across layers"): a multi-hour section used to
occupy a render-pool worker for its ENTIRE backlog (render_all_hours draining every
stale hour in one call) before the next queued section could start. _render_worker now
threads an optional max_hours through to run(), and start_scheduler() dispatches
multi-hour sections in rounds of one hour each instead of one all-hours call, so no
single section's backlog can monopolise the pool.
"""
import inspect
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atmos_gl.layer_builder import (
    MULTI_HOUR_SECTIONS,
    SINGLE_SHOT_SECTIONS,
    TASK_CLASSES,
    _render_worker,
    _updater_class,
    build_layer_channel_keys,
    LayerBuilder,
    workers_for_tier,
    dispatchable_sections,
)
from atmos_gl.round_robin_order import RoundRobinOrder


def test_multi_hour_and_single_shot_sections_partition_task_classes():
    assert set(MULTI_HOUR_SECTIONS) | set(SINGLE_SHOT_SECTIONS) == set(TASK_CLASSES)
    assert set(MULTI_HOUR_SECTIONS).isdisjoint(SINGLE_SHOT_SECTIONS)
    assert set(SINGLE_SHOT_SECTIONS) == {
        "sst", "clouds", "markers", "greenhouse_gases", "air_quality", "flood_risk",
    }
    assert set(MULTI_HOUR_SECTIONS) == {
        "isobars", "precipitation", "wind", "currents", "jetstream", "waves",
        "temperature", "ozone", "stormwatch", "pwat", "fires",
    }


def test_workers_for_tier_low_is_always_one():
    """Low is the one tier that actually guarantees no concurrent-process pile-up
    regardless of core count -- fully sequential, one worker at a time."""
    assert workers_for_tier("low", cpu_count=1) == 1
    assert workers_for_tier("low", cpu_count=4) == 1
    assert workers_for_tier("low", cpu_count=32) == 1


def test_workers_for_tier_medium_scales_with_cores_floored_at_two():
    assert workers_for_tier("medium", cpu_count=1) == 2
    assert workers_for_tier("medium", cpu_count=4) == 2
    assert workers_for_tier("medium", cpu_count=8) == 4


def test_workers_for_tier_high_matches_todays_unchanged_formula():
    """High must stay a complete no-op relative to LayerBuilder's pre-existing
    hardcoded formula: min(len(TASK_CLASSES), cpu_count or 4)."""
    assert workers_for_tier("high", cpu_count=2) == min(len(TASK_CLASSES), 2)
    assert workers_for_tier("high", cpu_count=100) == len(TASK_CLASSES)
    assert workers_for_tier("high", cpu_count=None) == min(len(TASK_CLASSES), 4)


def test_workers_for_tier_defaults_unknown_tier_to_medium():
    assert workers_for_tier("bogus", cpu_count=8) == workers_for_tier("medium", cpu_count=8)


def test_updater_class_unwraps_partial_bindings():
    from atmos_gl.tasks.scalar_field import ScalarFieldUpdater

    assert _updater_class(TASK_CLASSES["ozone"]) is ScalarFieldUpdater
    assert _updater_class(TASK_CLASSES["isobars"]) is TASK_CLASSES["isobars"]


def test_every_multi_hour_section_run_accepts_max_hours():
    """_render_worker calls run(max_hours=max_hours) unconditionally for every
    TASK_CLASSES entry (see test_render_worker_forwards_max_hours_and_returns_plotted_count
    above) -- but that test only exercises a fake updater, so it can't catch a REAL
    multi-hour section whose run() doesn't declare max_hours. That gap let
    PrecipitationUpdater.run() ship without the parameter: every dispatch round raised
    TypeError("PrecipitationUpdater.run() got an unexpected keyword argument
    'max_hours'"), so precipitation silently never rendered a new hour while every other
    multi-hour layer kept advancing."""
    for section in MULTI_HOUR_SECTIONS:
        cls = _updater_class(TASK_CLASSES[section])
        params = inspect.signature(cls.run).parameters
        accepts_max_hours = "max_hours" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        assert accepts_max_hours, (
            f"{cls.__name__}.run() (section {section!r}) doesn't accept max_hours; "
            f"_render_worker calls every section's run(max_hours=...) unconditionally"
        )


def test_render_worker_forwards_max_hours_and_returns_plotted_count():
    fake_updater = MagicMock()
    fake_updater.run.return_value = 1
    fake_cls = MagicMock(return_value=fake_updater)

    with patch.dict("atmos_gl.layer_builder.TASK_CLASSES", {"fake": fake_cls}), \
         patch("atmos_gl.layer_builder.AtmosGLConfig"), \
         patch("atmos_gl.layer_builder.MapData"):
        result = _render_worker("cfg.json", "fake", {}, max_hours=1)

    fake_updater.run.assert_called_once_with(max_hours=1)
    assert result == ("fake", None, 1)


def test_render_worker_reports_zero_plotted_on_none_return():
    """Single-shot layers' run() returns None -- _render_worker must not propagate
    that as the plotted count (the round-robin loop treats None like 0, but coercing
    it here keeps the (section, error, plotted) tuple shape uniform)."""
    fake_updater = MagicMock()
    fake_updater.run.return_value = None
    fake_cls = MagicMock(return_value=fake_updater)

    with patch.dict("atmos_gl.layer_builder.TASK_CLASSES", {"fake": fake_cls}), \
         patch("atmos_gl.layer_builder.AtmosGLConfig"), \
         patch("atmos_gl.layer_builder.MapData"):
        result = _render_worker("cfg.json", "fake", {})

    assert result == ("fake", None, 0)


def test_render_worker_catches_exceptions():
    fake_cls = MagicMock(side_effect=RuntimeError("boom"))

    with patch.dict("atmos_gl.layer_builder.TASK_CLASSES", {"fake": fake_cls}), \
         patch("atmos_gl.layer_builder.AtmosGLConfig"), \
         patch("atmos_gl.layer_builder.MapData"):
        section, error, plotted = _render_worker("cfg.json", "fake", {})

    assert section == "fake"
    assert "boom" in error
    assert plotted == 0


def testbuild_layer_channel_keys_maps_every_field_collector_product():
    class _FakeGfsAtmos:
        channel_key = "gfs_atmos"
        products = {"isobars": None, "wind": None, "humidity": None}

    class _FakeGfsWaves:
        channel_key = "gfs_waves"
        products = {"waves": None}

    mapping = build_layer_channel_keys((_FakeGfsAtmos, _FakeGfsWaves), ())

    assert mapping == {
        "isobars": "gfs_atmos",
        "wind": "gfs_atmos",
        "humidity": "gfs_atmos",
        "waves": "gfs_waves",
    }


def testbuild_layer_channel_keys_maps_cache_collectors_by_section():
    class _FakeSst:
        channel_key = "sst"
        section = "sst"

    mapping = build_layer_channel_keys((), (_FakeSst,))

    assert mapping == {"sst": "sst"}


def testbuild_layer_channel_keys_maps_by_settings_section_when_set():
    """The greenhouse_gases layer's two collectors (CamsGhgForecastCollector,
    CamsEgg4BaselineCollector) each have their OWN `section` (for independent
    _drive() scheduling/status rows) but share one settings_section
    ("greenhouse_gases", the actual TASK_CLASSES/layer key) -- the mapping must key
    off settings_section when set, or channel-disabling either collector would never
    gray out / block dispatch of the greenhouse_gases layer at all."""
    class _FakeForecast:
        channel_key = "ghg_forecast"
        section = "ghg_forecast"
        settings_section = "greenhouse_gases"

    mapping = build_layer_channel_keys((), (_FakeForecast,))

    assert mapping == {"greenhouse_gases": "ghg_forecast"}


def testbuild_layer_channel_keys_first_collector_wins_when_two_share_a_settings_section():
    """greenhouse_gases has two collectors sharing one settings_section -- the FIRST
    one registered (list order) wins the layer's channel mapping, deliberately the
    always-required one (CamsGhgForecastCollector), not whichever happens to be listed last."""
    class _FakeForecast:
        channel_key = "ghg_forecast"
        section = "ghg_forecast"
        settings_section = "greenhouse_gases"

    class _FakeEgg4:
        channel_key = "ghg_egg4_baseline"
        section = "ghg_egg4_baseline"
        settings_section = "greenhouse_gases"

    mapping = build_layer_channel_keys((), (_FakeForecast, _FakeEgg4))

    assert mapping == {"greenhouse_gases": "ghg_forecast"}


def testbuild_layer_channel_keys_skips_a_collector_with_no_channel_key():
    """markers isn't part of channel_enabled -- must not appear in the mapping at all
    (not even as None), since a `None` value would be indistinguishable from
    "channel_key wasn't set" if ever iterated rather than looked up by key."""
    class _FakeUngated:
        channel_key = None
        products = {"markers": None}
        section = "markers"

    field_mapping = build_layer_channel_keys((_FakeUngated,), ())
    cache_mapping = build_layer_channel_keys((), (_FakeUngated,))

    assert field_mapping == {}
    assert cache_mapping == {}


def test_dispatchable_sections_excludes_a_section_whose_channel_is_disabled():
    channel_enabled = {"rtofs_currents": False}
    layer_channel_keys = {"currents": "rtofs_currents", "jetstream": "gfs_atmos"}

    result = dispatchable_sections(channel_enabled, layer_channel_keys, ["currents", "jetstream"])

    assert result == ["jetstream"]


def test_dispatchable_sections_keeps_a_section_with_no_channel_mapping():
    """markers has no channel_key at all (not every layer maps to exactly one
    toggleable channel) -- must never be excluded, regardless of channel_enabled."""
    channel_enabled = {"gfs_atmos": False}
    layer_channel_keys = {"isobars": "gfs_atmos"}  # "markers" absent entirely

    result = dispatchable_sections(channel_enabled, layer_channel_keys, ["isobars", "markers"])

    assert result == ["markers"]


def test_dispatchable_sections_defaults_an_unwritten_channel_to_enabled():
    """Matches _serialize()'s existing convention: a channel_key not yet written to
    channel_enabled (e.g. right after upgrading to this feature) defaults to True."""
    result = dispatchable_sections({}, {"isobars": "gfs_atmos"}, ["isobars"])

    assert result == ["isobars"]


def test_dispatchable_sections_preserves_input_order():
    channel_enabled = {}
    layer_channel_keys = {}

    result = dispatchable_sections(channel_enabled, layer_channel_keys, ["waves", "isobars", "sst"])

    assert result == ["waves", "isobars", "sst"]


def make_bare_layer_builder():
    lb = LayerBuilder.__new__(LayerBuilder)
    lb.process_status_adapter = MagicMock()
    # Permissive defaults (no channels disabled, no channel mapping at all) so tests
    # unconcerned with channel-aware dispatch keep dispatching every section, same as
    # before that feature existed. Tests that DO care override .config/
    # ._layer_channel_keys themselves.
    lb.config = MagicMock()
    lb.config.get_setting.return_value = {}
    lb._layer_channel_keys = {}
    lb.order = RoundRobinOrder(MULTI_HOUR_SECTIONS)
    lb._last_dispatch_ts = {}
    return lb


def _stub_config(performance_tier, enabled=True, log_level=None):
    """A config double whose get_setting() answers exactly the three calls
    refresh_settings() makes -- (layer_builder, enabled), (common, log_level),
    (common, performance_tier) -- regardless of call order or default arg presence."""
    values = {
        ("layer_builder", "enabled"): enabled,
        ("common", "log_level"): log_level,
        ("common", "performance_tier"): performance_tier,
    }
    cfg = MagicMock()
    cfg.get_setting.side_effect = lambda section, key, default=None: values.get(
        (section, key), default
    )
    return cfg


def test_refresh_settings_recreates_the_pool_when_the_tier_changes():
    lb = make_bare_layer_builder()
    lb.config = _stub_config(performance_tier="low")
    lb._tier = "medium"
    lb._max_workers = 2
    old_pool = MagicMock()
    lb._pool = old_pool
    new_pool = MagicMock()
    lb._new_pool = MagicMock(return_value=new_pool)

    lb.refresh_settings()

    old_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    lb._new_pool.assert_called_once()
    assert lb._pool is new_pool
    assert lb._tier == "low"
    assert lb._max_workers == workers_for_tier("low", os.cpu_count())


def test_refresh_settings_leaves_the_pool_alone_when_the_tier_is_unchanged():
    lb = make_bare_layer_builder()
    lb.config = _stub_config(performance_tier="medium")
    lb._tier = "medium"
    lb._max_workers = workers_for_tier("medium", os.cpu_count())
    old_pool = MagicMock()
    lb._pool = old_pool
    lb._new_pool = MagicMock()

    lb.refresh_settings()

    old_pool.shutdown.assert_not_called()
    lb._new_pool.assert_not_called()
    assert lb._pool is old_pool


def test_refresh_settings_updates_max_workers_without_touching_the_pool_before_one_exists():
    """The very first refresh_settings() call inside start_scheduler() runs before
    self._pool has been created -- a tier change detected then must not try to shut
    down a pool that doesn't exist yet."""
    lb = make_bare_layer_builder()
    lb.config = _stub_config(performance_tier="low")
    lb._tier = "medium"
    lb._max_workers = 2
    lb._pool = None
    lb._new_pool = MagicMock()

    lb.refresh_settings()

    lb._new_pool.assert_not_called()
    assert lb._pool is None
    assert lb._tier == "low"
    assert lb._max_workers == workers_for_tier("low", os.cpu_count())


def test_handle_results_reports_plotted_count_per_section():
    lb = make_bare_layer_builder()
    results = [("isobars", None, 3), ("pwat", None, 0), ("sst", None, 0)]

    broken, plotted_by_section = lb._handle_results(["isobars", "pwat", "sst"], results)

    assert broken is False
    assert plotted_by_section == {"isobars": 3, "pwat": 0, "sst": 0}


def test_handle_results_excludes_failed_sections_from_plotted():
    lb = make_bare_layer_builder()
    results = [("isobars", "RuntimeError('boom')", 0), ("pwat", None, 2)]

    broken, plotted_by_section = lb._handle_results(["isobars", "pwat"], results)

    assert broken is False
    assert "isobars" not in plotted_by_section
    assert plotted_by_section == {"pwat": 2}


def test_handle_results_still_detects_a_broken_pool():
    from concurrent.futures.process import BrokenProcessPool

    lb = make_bare_layer_builder()
    results = [BrokenProcessPool("worker died")]

    broken, plotted_by_section = lb._handle_results(["isobars"], results)

    assert broken is True
    assert plotted_by_section == {}


@pytest.mark.asyncio
async def test_run_dispatch_cycle_redispatches_single_shot_sections_every_round():
    """Round 1 dispatches every section (single-shot + all multi-hour). Round 2 still
    includes every single-shot section -- they ride along on every round rather than
    being starved behind however long the multi-hour backlog takes to drain (issue
    #240) -- plus only the multi-hour sections that still had a backlog last round; a
    multi-hour section with nothing left to render drops out just as quickly as one
    that never had anything pending."""
    lb = make_bare_layer_builder()
    round_1 = {s: 1 for s in ("isobars", "precipitation", "wind", "currents", "waves",
                               "temperature", "ozone", "stormwatch")}
    round_1["pwat"] = 0  # never had a backlog
    round_2 = {s: 0 for s in round_1 if round_1[s] > 0}  # everyone catches up in round 2
    lb._dispatch_round = AsyncMock(side_effect=[round_1, round_2])

    await lb._run_dispatch_cycle(loop=MagicMock(), baseline={})

    assert lb._dispatch_round.call_count == 2
    round_1_sections = set(lb._dispatch_round.call_args_list[0].args[1])
    round_2_sections = set(lb._dispatch_round.call_args_list[1].args[1])
    assert round_1_sections == set(SINGLE_SHOT_SECTIONS) | set(MULTI_HOUR_SECTIONS)
    assert round_2_sections == set(SINGLE_SHOT_SECTIONS) | (set(round_1) - {"pwat"})


@pytest.mark.asyncio
async def test_run_dispatch_cycle_never_starves_single_shot_sections_behind_a_long_backlog():
    """Regression guard for issue #240, found live: a large multi-hour backlog (many
    rounds, each advancing only 1 hour per section) used to starve single-shot
    sections -- greenhouse_gases/air_quality picked up a config change 30+ minutes
    late because they were dropped from `pending` after round 1 and _run_dispatch_cycle
    doesn't return (letting the outer scheduler loop re-dispatch them) until the ENTIRE
    multi-hour backlog drains. Every single-shot section must appear in every round,
    however many rounds a slow backlog takes to clear."""
    lb = make_bare_layer_builder()
    # isobars alone takes 5 rounds to drain; every OTHER multi-hour section already has
    # nothing pending from round 1 onward (drops out immediately, as normal).
    rounds = [{"isobars": 1} for _ in range(4)] + [{"isobars": 0}]
    for r in rounds[:-1]:
        for s in MULTI_HOUR_SECTIONS:
            r.setdefault(s, 0)
    lb._dispatch_round = AsyncMock(side_effect=rounds)

    await lb._run_dispatch_cycle(loop=MagicMock(), baseline={})

    assert lb._dispatch_round.call_count == 5
    for call in lb._dispatch_round.call_args_list:
        dispatched = set(call.args[1])
        assert set(SINGLE_SHOT_SECTIONS) <= dispatched, (
            "single-shot sections must be present in every round, not just the first"
        )


@pytest.mark.asyncio
async def test_run_dispatch_cycle_skips_a_section_whose_channel_is_disabled():
    """A section whose backing channel is manually disabled never gets its worker
    process spawned at all -- not even once, and not just dropped after round 1."""
    lb = make_bare_layer_builder()
    cfg = MagicMock()
    cfg.get_setting.side_effect = lambda section, key, default=None: (
        {"rtofs_currents": False}
        if (section, key) == ("data_collector", "channel_enabled")
        else default
    )
    lb.config = cfg
    lb._layer_channel_keys = {"currents": "rtofs_currents"}
    lb._dispatch_round = AsyncMock(return_value={})

    await lb._run_dispatch_cycle(loop=MagicMock(), baseline={})

    lb._dispatch_round.assert_called_once()
    dispatched_sections = set(lb._dispatch_round.call_args_list[0].args[1])
    assert "currents" not in dispatched_sections
    assert dispatched_sections == (
        (set(SINGLE_SHOT_SECTIONS) | set(MULTI_HOUR_SECTIONS)) - {"currents"}
    )


@pytest.mark.asyncio
async def test_run_dispatch_cycle_dispatches_everything_when_no_channel_mapping_is_set():
    """A bare LayerBuilder with no _layer_channel_keys computed yet (or an empty
    mapping) must not accidentally exclude every section -- absence of a mapping means
    "not applicable", not "disabled"."""
    lb = make_bare_layer_builder()
    cfg = MagicMock()
    cfg.get_setting.return_value = {}
    lb.config = cfg
    lb._layer_channel_keys = {}
    lb._dispatch_round = AsyncMock(return_value={})

    await lb._run_dispatch_cycle(loop=MagicMock(), baseline={})

    dispatched_sections = set(lb._dispatch_round.call_args_list[0].args[1])
    assert dispatched_sections == set(SINGLE_SHOT_SECTIONS) | set(MULTI_HOUR_SECTIONS)


@pytest.mark.asyncio
async def test_run_dispatch_cycle_dispatches_multi_hour_sections_in_the_orders_priority_order():
    """The whole point of RoundRobinOrder: a section bumped to the front via
    reorder() (e.g. through POST /api/layer_builder/priority) is the first multi-hour
    section in the very next round's dispatch list -- not just present in the set."""
    lb = make_bare_layer_builder()
    lb.order.reorder(["pwat"])
    lb._dispatch_round = AsyncMock(return_value={s: 0 for s in MULTI_HOUR_SECTIONS})

    await lb._run_dispatch_cycle(loop=MagicMock(), baseline={})

    sections = lb._dispatch_round.call_args_list[0].args[1]
    multi_hour_dispatched = [s for s in sections if s in MULTI_HOUR_SECTIONS]
    assert multi_hour_dispatched[0] == "pwat"


@pytest.mark.asyncio
async def test_run_dispatch_cycle_stops_once_nothing_reports_progress():
    lb = make_bare_layer_builder()
    lb._dispatch_round = AsyncMock(return_value={s: 0 for s in MULTI_HOUR_SECTIONS})

    await lb._run_dispatch_cycle(loop=MagicMock(), baseline={})

    lb._dispatch_round.assert_called_once()  # round 1 only -- nothing had a backlog


# --- Watchdog: a multi-hour section silently dropping out of round-robin dispatch
# (found live 2026-08-10/11 -- isobars stopped being dispatched for ~27h with no
# error, no crash, while every other multi-hour section kept rendering normally) ---


def test_handle_results_records_dispatch_timestamp_on_success():
    lb = make_bare_layer_builder()

    with patch("atmos_gl.layer_builder.time.time", return_value=1000.0):
        lb._handle_results(["isobars"], [("isobars", None, 1)])

    assert lb._last_dispatch_ts == {"isobars": 1000.0}


def test_handle_results_records_dispatch_timestamp_on_a_failed_task():
    """A section erroring is still proof the round-robin loop is actively dispatching
    it -- the failure mode this guards against is a section going silently MISSING
    from dispatch entirely, not one that dispatches and fails."""
    lb = make_bare_layer_builder()

    with patch("atmos_gl.layer_builder.time.time", return_value=1000.0):
        lb._handle_results(["isobars"], [("isobars", "RuntimeError('boom')", 0)])

    assert lb._last_dispatch_ts == {"isobars": 1000.0}


def test_handle_results_records_dispatch_timestamp_on_a_broken_pool():
    from concurrent.futures.process import BrokenProcessPool

    lb = make_bare_layer_builder()

    with patch("atmos_gl.layer_builder.time.time", return_value=1000.0):
        lb._handle_results(["isobars"], [BrokenProcessPool("worker died")])

    assert lb._last_dispatch_ts == {"isobars": 1000.0}


def test_handle_results_does_not_track_single_shot_sections():
    """Single-shot sections (sst, clouds, ...) have no per-hour dispatch concept and
    aren't part of the round-robin this watchdog protects -- must never appear in
    _last_dispatch_ts at all, not even as a harmless extra entry."""
    lb = make_bare_layer_builder()

    with patch("atmos_gl.layer_builder.time.time", return_value=1000.0):
        lb._handle_results(["sst", "isobars"], [("sst", None, 0), ("isobars", None, 1)])

    assert lb._last_dispatch_ts == {"isobars": 1000.0}


def test_check_watchdog_does_nothing_when_every_section_is_recent():
    lb = make_bare_layer_builder()
    lb._last_dispatch_ts = {"isobars": 1000.0, "wind": 1000.0}

    with patch("atmos_gl.layer_builder.time.time", return_value=1000.0 + 60), \
         patch("atmos_gl.layer_builder.os._exit") as exit_mock:
        lb._check_watchdog()

    exit_mock.assert_not_called()


def test_check_watchdog_ignores_a_section_never_dispatched_yet():
    """A section absent from _last_dispatch_ts (e.g. right after a fresh restart, or
    one still waiting its turn in a large backlog) must never be flagged -- only a
    section that WAS seen and then stopped updating counts as stale."""
    lb = make_bare_layer_builder()
    lb._last_dispatch_ts = {}

    with patch("atmos_gl.layer_builder.time.time", return_value=1_000_000.0), \
         patch("atmos_gl.layer_builder.os._exit") as exit_mock:
        lb._check_watchdog()

    exit_mock.assert_not_called()


def test_check_watchdog_does_not_fire_just_under_the_threshold():
    from atmos_gl.layer_builder import WATCHDOG_STALE_SECONDS

    lb = make_bare_layer_builder()
    lb._last_dispatch_ts = {"isobars": 1000.0}

    with patch("atmos_gl.layer_builder.time.time", return_value=1000.0 + WATCHDOG_STALE_SECONDS - 1), \
         patch("atmos_gl.layer_builder.os._exit") as exit_mock:
        lb._check_watchdog()

    exit_mock.assert_not_called()


def test_check_watchdog_forces_exit_once_a_section_exceeds_the_threshold():
    from atmos_gl.layer_builder import WATCHDOG_STALE_SECONDS

    lb = make_bare_layer_builder()
    lb._last_dispatch_ts = {"isobars": 1000.0, "wind": 1000.0 + WATCHDOG_STALE_SECONDS}

    with patch("atmos_gl.layer_builder.time.time", return_value=1000.0 + WATCHDOG_STALE_SECONDS + 1), \
         patch("atmos_gl.layer_builder.os._exit") as exit_mock:
        lb._check_watchdog()

    exit_mock.assert_called_once_with(1)
