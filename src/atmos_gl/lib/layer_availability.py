#!/usr/bin/env python3
"""Computes, per Show-tab layer section, whether its data is actually being collected
right now -- distinct from that layer's own `enabled` flag, which (per
collectors/__init__.py's module docstring) is purely a frontend-visibility control for
most layers; collection itself keeps running unconditionally of it. Three independent
gates can silence real collection instead:

  1. data_collector.enabled (the master switch) -- governs the synchronous event-feed/
     file-cache/field-collector families (quakes, storms, volcanoes, fires, satellites,
     markers, sst, clouds, greenhouse_gases, air_quality, and the gfs_atmos/gfs_waves/
     rtofs_currents-derived layers).
  2. data_collector.channel_enabled[channel_key] -- a per-channel override within that
     same master-switch-gated family (see collectors/driving.py's CollectorDriver.drive).
  3. <section>_collector.enabled -- the three async collectors' (shipping, lightning,
     flightradar) own, separate kill-switch, independent of the master switch (see
     collectors/service.py's _spawn_embedded_collectors, which isn't gated by `enabled`
     at all -- only each embedded collector's own run() loop is).

Backs the public GET /api/layer_availability route (routes/layer_availability.py):
consumed by config.html/me_settings.html (gray out + explain a layer whose collector is
off) and the live globe's _reconcile.js (guards mount/unmount on this too, so a layer
stops rendering within one poll interval of the relevant switch flipping -- no page
reload needed). A section absent from the result has no collector dependency at all
(e.g. terminator, landmass -- pure client-side) and should be treated as always
collecting.
"""
from atmos_gl.collectors import COLLECTORS, CACHE_COLLECTORS, FIELD_COLLECTOR_CLASSES
from atmos_gl.layer_builder import build_layer_channel_keys
from atmos_gl.lib.troublespot_contours import MIN_CONVERGENCE_TYPES

REASON_PAUSED = (
    "Data collection is currently paused by the site admin — this layer won't show "
    "data until it's resumed."
)

# section -> its collector's OWN config section, for the three async collectors that
# keep a real, separate kill-switch rather than a channel_key (see module docstring
# point 3). Independent of the two dicts below.
_ASYNC_COLLECTOR_SECTIONS = {
    "shipping": "shipping_collector",
    "lightning": "lightning_collector",
    "flightradar": "flightradar_collector",
}

# Troublespots (issue #366) is a derived view over these four collectors' channels,
# not a single-channel-gated layer -- it requires AT LEAST MIN_CONVERGENCE_TYPES of
# them enabled to ever produce a result (fewer than that, and its own 2+-type
# convergence rule can never be satisfied). Reuses the SAME threshold constant the
# contour-banding logic itself uses (lib/troublespot_contours.py), rather than a
# second hardcoded "2" that could silently drift out of sync with it.
_TROUBLESPOTS_CHANNELS = ("quakes", "fires", "volcanoes", "world_events")


def _channel_gated_sections() -> dict:
    """section -> channel_key for every layer gated by data_collector.channel_enabled.

    Two families: build_layer_channel_keys() already maps the TASK_CLASSES-backed
    layers (isobars, sst, ...); the event-feed COLLECTORS (quakes, storms, volcanoes,
    fires, satellites) aren't TASK_CLASSES entries (no render Updater) so aren't
    covered by it, but each one's own `channel_key` class attribute IS the frontend
    layer's Show-tab section name (e.g. SatellitesCollector.section is
    "satellites_collector", a distinct settings-only section, but its channel_key is
    "satellites" -- the actual layer). markers has channel_key=None (reads a local
    file, no "good citizen" opt-out to offer -- see CollectorBase.channel_key) and is
    handled separately, gated by the master switch alone."""
    mapping = dict(build_layer_channel_keys(FIELD_COLLECTOR_CLASSES, CACHE_COLLECTORS))
    for cls in COLLECTORS:
        channel_key = getattr(cls, "channel_key", None)
        if channel_key:
            mapping[channel_key] = channel_key
    return mapping


def _entry(collecting: bool) -> dict:
    return {"collecting": collecting, "reason": None if collecting else REASON_PAUSED}


def compute_layer_availability(config) -> dict:
    """{section: {"collecting": bool, "reason": str | None}} for every layer section
    with a real collection dependency."""
    data_collector_enabled = config.section_enabled("data_collector")
    channel_enabled = config.get_setting("data_collector", "channel_enabled", {}) or {}

    result = {}
    for section, channel_key in _channel_gated_sections().items():
        result[section] = _entry(data_collector_enabled and channel_enabled.get(channel_key, True))

    # markers: no channel_key, gated by the master switch alone.
    result["markers"] = _entry(data_collector_enabled)

    for section, collector_section in _ASYNC_COLLECTOR_SECTIONS.items():
        result[section] = _entry(config.section_enabled(collector_section))

    enabled_channel_count = sum(
        1 for k in _TROUBLESPOTS_CHANNELS if channel_enabled.get(k, True)
    )
    result["troublespots"] = _entry(
        data_collector_enabled and enabled_channel_count >= MIN_CONVERGENCE_TYPES
    )

    return result
