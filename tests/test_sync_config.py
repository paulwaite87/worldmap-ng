#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "tools"))
from sync_config import sync_shape


def test_adds_a_missing_top_level_section():
    live = {"fires": {"enabled": True}}
    template = {"fires": {"enabled": False}, "world_events": {"enabled": False, "opacity": 80}}

    synced, added, removed = sync_shape(live, template)

    assert synced["world_events"] == {"enabled": False, "opacity": 80}
    assert added == ["world_events"]
    assert removed == []


def test_preserves_existing_values_it_does_not_overwrite_with_template_defaults():
    live = {"fires": {"enabled": True, "opacity": 55}}
    template = {"fires": {"enabled": False, "opacity": 70}}

    synced, added, removed = sync_shape(live, template)

    assert synced == {"fires": {"enabled": True, "opacity": 55}}
    assert added == []
    assert removed == []


def test_removes_a_section_the_template_no_longer_defines():
    live = {"fires": {"enabled": True}, "deprecated_layer": {"enabled": True}}
    template = {"fires": {"enabled": False}}

    synced, added, removed = sync_shape(live, template)

    assert synced == {"fires": {"enabled": True}}
    assert removed == ["deprecated_layer"]
    assert added == []


def test_recurses_into_nested_dicts_like_data_collector_datasources():
    live = {
        "data_collector": {
            "datasources": {"gfs": "https://example.com/gfs"},
            "channel_enabled": {"gfs_atmos": True, "old_channel": True},
        }
    }
    template = {
        "data_collector": {
            "datasources": {"gfs": "https://default.example/gfs", "world_events": "https://gdelt.example"},
            "channel_enabled": {"gfs_atmos": True},
        }
    }

    synced, added, removed = sync_shape(live, template)

    assert synced["data_collector"]["datasources"] == {
        "gfs": "https://example.com/gfs",
        "world_events": "https://gdelt.example",
    }
    assert synced["data_collector"]["channel_enabled"] == {"gfs_atmos": True}
    assert added == ["data_collector.datasources.world_events"]
    assert removed == ["data_collector.channel_enabled.old_channel"]


def test_leaves_a_list_value_untouched_even_though_the_key_exists_in_both():
    live = {"satellites": {"sat_names": ["ISS (ZARYA)", "NOAA 19"]}}
    template = {"satellites": {"sat_names": ["ISS (ZARYA)", "HST"]}}

    synced, added, removed = sync_shape(live, template)

    assert synced["satellites"]["sat_names"] == ["ISS (ZARYA)", "NOAA 19"]
    assert added == []
    assert removed == []


def test_no_changes_when_already_in_sync():
    live = {"fires": {"enabled": True}}
    template = {"fires": {"enabled": False}}

    synced, added, removed = sync_shape(live, template)

    assert synced == live
    assert added == []
    assert removed == []
