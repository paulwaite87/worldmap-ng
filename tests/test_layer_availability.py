#!/usr/bin/env python3
"""Tests for compute_layer_availability (lib/layer_availability.py) -- the composite
"is this layer's data actually being collected right now" signal, distinct from a
layer's own frontend-visibility `enabled` flag (see collectors/__init__.py's module
docstring: for most layers `enabled` is purely cosmetic, collection is unconditional).

Grilled design (in-conversation, no tracked issue): three independent gates can
silence a layer's real data flow --
  1. data_collector.enabled (master switch) -- governs the synchronous event-feed/
     file-cache/field-collector families (quakes, storms, volcanoes, fires,
     satellites, markers, sst, clouds, greenhouse_gases, air_quality, and the
     gfs_atmos/gfs_waves/rtofs_currents-derived layers).
  2. data_collector.channel_enabled[channel_key] -- a per-channel override within
     that same master-switch-gated family.
  3. <section>_collector.enabled -- the three async collectors' (shipping, lightning,
     flightradar) own, separate kill-switch, independent of the master switch.
Layers with no collector dependency at all (terminator, landmass, ...) are absent
from the result entirely -- the frontend/consumer treats a missing key as always
collecting.
"""
from atmos_gl.lib.layer_availability import compute_layer_availability, REASON_PAUSED


class FakeConfig:
    """Minimal AtmosGLConfig stand-in: only section_enabled()/get_setting(), the two
    methods compute_layer_availability actually calls."""

    def __init__(self, config: dict):
        self.config = config

    def section_enabled(self, section):
        return bool(self.config.get(section, {}).get("enabled", False))

    def get_setting(self, section, setting, default=None):
        return self.config.get(section, {}).get(setting, default)


def _base_config(**overrides) -> dict:
    config = {
        "data_collector": {
            "enabled": True,
            "channel_enabled": {
                "gfs_atmos": True, "gfs_waves": True, "rtofs_currents": True,
                "sst": True, "clouds": True, "ghg_forecast": True,
                "ghg_egg4_baseline": True, "air_quality": True,
                "quakes": True, "volcanoes": True, "fires": True,
                "satellites": True, "storms": True,
            },
        },
        "shipping_collector": {"enabled": True},
        "lightning_collector": {"enabled": True},
        "flightradar_collector": {"enabled": True},
    }
    config.update(overrides)
    return config


def test_every_channel_gated_layer_collects_when_everything_is_on():
    availability = compute_layer_availability(FakeConfig(_base_config()))
    for section in (
        "isobars", "wind", "jetstream", "precipitation", "pwat", "temperature",
        "ozone", "stormwatch", "waves", "currents", "sst", "clouds",
        "greenhouse_gases", "air_quality", "quakes", "storms", "volcanoes",
        "fires", "satellites",
    ):
        assert availability[section] == {"collecting": True, "reason": None}, section


def test_master_switch_off_silences_every_channel_gated_layer():
    config = _base_config()
    config["data_collector"]["enabled"] = False
    availability = compute_layer_availability(FakeConfig(config))

    for section in ("isobars", "sst", "quakes", "markers", "greenhouse_gases"):
        assert availability[section]["collecting"] is False
        assert availability[section]["reason"] == REASON_PAUSED

    # The three async collectors are NOT gated by the master switch at all.
    for section in ("shipping", "lightning", "flightradar"):
        assert availability[section]["collecting"] is True


def test_a_single_channel_disabled_only_silences_layers_fed_by_that_channel():
    config = _base_config()
    config["data_collector"]["channel_enabled"]["gfs_atmos"] = False
    availability = compute_layer_availability(FakeConfig(config))

    assert availability["isobars"]["collecting"] is False
    assert availability["wind"]["collecting"] is False
    # A sibling GFS-derived product fed by a different channel stays on.
    assert availability["waves"]["collecting"] is True
    assert availability["sst"]["collecting"] is True


def test_markers_is_gated_by_the_master_switch_only_no_channel_key():
    config = _base_config()
    # markers has no channel_enabled entry at all -- confirm it's still silenced by
    # the master switch, and stays on when the master switch is on regardless of
    # what's in channel_enabled.
    availability_on = compute_layer_availability(FakeConfig(config))
    assert availability_on["markers"] == {"collecting": True, "reason": None}

    config["data_collector"]["enabled"] = False
    availability_off = compute_layer_availability(FakeConfig(config))
    assert availability_off["markers"]["collecting"] is False


def test_a_missing_channel_enabled_entry_defaults_to_on():
    config = _base_config()
    del config["data_collector"]["channel_enabled"]["quakes"]
    availability = compute_layer_availability(FakeConfig(config))
    assert availability["quakes"]["collecting"] is True


def test_async_collectors_are_gated_by_their_own_collector_section_only():
    config = _base_config()
    config["flightradar_collector"]["enabled"] = False
    # The master switch stays on -- proves flightradar isn't affected by it.
    availability = compute_layer_availability(FakeConfig(config))

    assert availability["flightradar"] == {"collecting": False, "reason": REASON_PAUSED}
    assert availability["shipping"]["collecting"] is True
    assert availability["lightning"]["collecting"] is True


def test_async_collector_stays_off_even_when_the_master_switch_is_also_off():
    config = _base_config()
    config["data_collector"]["enabled"] = False
    config["shipping_collector"]["enabled"] = False
    availability = compute_layer_availability(FakeConfig(config))
    assert availability["shipping"]["collecting"] is False


def test_sections_with_no_collector_dependency_are_absent_from_the_result():
    availability = compute_layer_availability(FakeConfig(_base_config()))
    assert "terminator" not in availability
    assert "landmass" not in availability


# Troublespots (issue #366): requires >= MIN_CONVERGENCE_TYPES (2) of its 4 source
# collectors' channels enabled to ever produce a result -- not a single-channel gate.
def test_troublespots_collects_when_all_four_source_channels_are_on():
    availability = compute_layer_availability(FakeConfig(_base_config()))
    assert availability["troublespots"] == {"collecting": True, "reason": None}


def test_troublespots_collects_with_exactly_the_minimum_two_channels_on():
    config = _base_config()
    config["data_collector"]["channel_enabled"]["fires"] = False
    config["data_collector"]["channel_enabled"]["volcanoes"] = False
    # quakes and world_events remain on -- exactly MIN_CONVERGENCE_TYPES (2).
    availability = compute_layer_availability(FakeConfig(config))
    assert availability["troublespots"]["collecting"] is True


def test_troublespots_stops_collecting_with_fewer_than_the_minimum_channels_on():
    config = _base_config()
    config["data_collector"]["channel_enabled"]["fires"] = False
    config["data_collector"]["channel_enabled"]["volcanoes"] = False
    config["data_collector"]["channel_enabled"]["world_events"] = False
    # Only quakes remains on -- below MIN_CONVERGENCE_TYPES (2).
    availability = compute_layer_availability(FakeConfig(config))
    assert availability["troublespots"] == {"collecting": False, "reason": REASON_PAUSED}


def test_troublespots_stops_collecting_when_the_master_switch_is_off():
    config = _base_config()
    config["data_collector"]["enabled"] = False
    availability = compute_layer_availability(FakeConfig(config))
    assert availability["troublespots"]["collecting"] is False


def test_troublespots_channels_missing_from_channel_enabled_default_to_on():
    # world_events has no channel_enabled entry at all in the base fixture already;
    # dropping quakes/fires/volcanoes too leaves every troublespots channel missing,
    # and a missing entry defaults to on (matching every other channel_key's default).
    config = _base_config()
    for key in ("quakes", "fires", "volcanoes"):
        del config["data_collector"]["channel_enabled"][key]
    availability = compute_layer_availability(FakeConfig(config))
    assert availability["troublespots"]["collecting"] is True
