#!/usr/bin/env python3
"""GET /api/data_status — collector + layer-task status for the Config UI's Data Status
tab. Constructs one lightweight, throwaway instance per collector/task class (same pattern
_drive()/_render_worker() already use) and calls its read-only data_status()/layer_status(),
never collect()/render(). Safe to call from map_api, which never runs collection itself —
that's the whole point of process_status being written by the orchestration layer and read
here independently.
"""
import logging

from fastapi import APIRouter, HTTPException, Depends

from atmos_gl.db.field_catalog_adapter import FieldCatalogAdapter
from atmos_gl.db.process_status_adapter import ProcessStatusAdapter
from atmos_gl.lib import fieldstore
from atmos_gl.lib.data_status import RUNS_PER_DAY_CHOICES
from atmos_gl.routes.auth import require_admin
from atmos_gl.routes.config import load_config

from atmos_gl.collectors import (
    COLLECTORS,
    CACHE_COLLECTORS,
    FIELD_COLLECTOR_CLASSES,
    EMBEDDABLE_COLLECTORS,
    resolve_embeddable,
)

from atmos_gl.layer_builder import TASK_CLASSES, build_layer_channel_keys
from atmos_gl.tasks.common import MapData
from atmos_gl.routes.field_specs import section_label

logger = logging.getLogger("atmos_gl.routes.status")
router = APIRouter(prefix="/api", tags=["Data Status"])

def _display_name(key: str) -> str:
    """Friendly name for a Data Status row -- matches the Show tab's wording exactly
    for anything that's a real config section (via section_label()). status_name-only
    identities that aren't real config sections (the 3 field collectors: "gfs_atmos",
    "gfs_waves", "rtofs_currents") would get section_label()'s .title() fallback here
    ("Gfs Atmos", not "GFS Atmos") -- they instead carry their own `display_label`
    class attribute (see GfsAtmosCollector etc.), resolved in _collect_status_rows()
    and passed to _serialize(), which prefers it over this function entirely."""
    return section_label(key)


# Sections whose runs_per_day is edited via this page's per-row widget instead of their
# own Settings tab (see _field_macros.html's exclusion list). data_collector isn't a
# section with its own row -- its runs_per_day is attached to the gfs_atmos row
# specifically in get_data_status() below, since that's the one field-collector row
# with real percent/next_update math already derived from the shared cadence.
RUNS_PER_DAY_SECTIONS = {
    "quakes", "volcanoes", "storms", "fires", "markers", "sst",
    "clouds", "satellites_collector", "world_events", "air_quality",
}


def _source_url(instance) -> str | None:
    """instance.source_url() if the collector defines it (every real CollectorBase/
    AsyncCollectorBase subclass does, see collectors/base.py) -- layer tasks and any
    pre-this-feature test stub simply lack the attribute, so this reads as None for
    them the same way getattr(CollectorCls, "channel_key", None) already does above."""
    fn = getattr(instance, "source_url", None)
    return fn() if callable(fn) else None


def _serialize(
    status: dict,
    channel_key: str | None,
    channel_enabled: dict,
    runs_per_day: int | None = None,
    source_url: str | None = None,
    runs_per_day_section: str | None = None,
    display_label: str | None = None,
) -> dict:
    """JSON-safe copy of a data_status()/layer_status() dict: datetimes -> ISO 8601.

    channel_key (data_collector.channel_enabled's key, e.g. "gfs_atmos") and the
    channel's current on/off state are attached here rather than inside
    data_status()/layer_status() themselves -- it's a Data Status UI concern (which
    row to gray out, and what the opt-out switch should show), not something those
    methods otherwise need to know. channel_key is None for a row that isn't gated by
    channel_enabled at all (e.g. markers, shipping/lightning -- see
    CollectorBase.channel_key); channel_on is then also None (not applicable) rather
    than True, so the frontend can distinguish "no switch" from "switch, currently on".
    Defaults to True (matches the wired-in gating's own default) when a key hasn't
    been written to channel_enabled yet, e.g. right after upgrading to this feature.

    Also adds display_name, a friendly version of `name` for the UI -- `name` itself
    is left untouched (some existing tests assert it verbatim, and it's still the
    right value for anything keying off the raw section/status_name). display_label
    (a collector class's own override, e.g. GfsAtmosCollector.display_label = "GFS
    Atmos") wins over the generic _display_name() derivation when given -- see that
    function's docstring for why the 3 field collectors need one.

    runs_per_day is the row's current cadence value if it's edited via this page's
    widget (see RUNS_PER_DAY_SECTIONS), else None so the frontend renders no widget.
    runs_per_day_section is the config section a cadence change actually saves to --
    almost always the row's own `name`, except the gfs_atmos row (shared cadence with
    gfs_waves/rtofs_currents, saved under "data_collector") -- attached here instead of
    left for the frontend to re-derive from `name`, architecture review candidate "one
    source of truth for which rows get the cadence widget".

    source_url is the collector's upstream URL (see CollectorBase.source_url()), so the
    Data Status page can render the row's label as a link to the live source data --
    None for layer rows (they don't fetch anything themselves) and any collector with
    no single browsable URL (e.g. markers)."""
    out = dict(status)
    for key in ("last_updated", "next_update"):
        v = out.get(key)
        if v is not None:
            out[key] = v.isoformat()
    out["channel_key"] = channel_key
    out["channel_on"] = channel_enabled.get(channel_key, True) if channel_key else None
    out["display_name"] = display_label or _display_name(out["name"])
    out["runs_per_day"] = runs_per_day
    out["runs_per_day_section"] = runs_per_day_section
    out["source_url"] = source_url
    return out


def _flightradar_hotspot_status(config, channel_enabled: dict, process_status_adapter) -> dict:
    """Layers-panel row for Flight Radar's hotspot-coverage progress (issue #215) --
    "N of M currently-prioritized cells have data yet", written each tick by
    AircraftCollector (collectors/aircraft.py's HOTSPOT_STATUS_NAME) via
    ProcessStatusAdapter.record_progress(). Not a TASK_CLASSES entry: Flight Radar has
    no render Updater (it's a direct DB->GeoJSON layer, not a rendered raster), so this
    is built by hand rather than forced through TaskCls(...).layer_status().

    segments (one {"queried": bool} per currently-prioritized cell) reuses the exact
    same segmented-progress-bar rendering config.html's GFS layers already have --
    see renderStatusTrack()'s "queried" branch. Cell IDENTITY isn't preserved here
    (only the count), so segments are ordered queried-first purely for a left-to-right
    fill effect, not because any specific segment corresponds to a specific cell."""
    row = process_status_adapter.get_process_status("flightradar_hotspot") or {}
    current = row.get("progress_current") or 0
    total = row.get("progress_total") or 0
    percent = 100.0 * current / total if total else 0.0
    segments = (
        [{"queried": i < current} for i in range(total)] if total else None
    )
    detail = (
        f"{current}/{total} hotspot cell(s) populated" if total
        else "No active viewport"
    )
    status = {
        "name": "flightradar_hotspot",
        "kind": "layer",
        "percent": round(percent, 1),
        "last_updated": row.get("updated_at"),
        "next_update": None,
        "enabled": bool(config.get_setting("flightradar", "enabled", False)),
        "detail": detail,
        "status": None,
        "health": None,
        "health_detail": None,
        "segments": segments,
        "run_date": None,
        "run_id": None,
    }
    return _serialize(status, None, channel_enabled, display_label="Flight Radar Hotspot")


def _collect_status_rows(
    classes,
    channel_enabled: dict,
    *,
    construct,
    cadence_of=lambda cls: (None, None),
) -> list[dict]:
    """Iterate `classes`, constructing each via construct(cls) and calling its
    data_status(), logging (not raising) a per-class failure so one bad collector can't
    blank the whole Data Status page -- the shared shape behind all three of
    get_data_status()'s collector loops (COLLECTORS+CACHE_COLLECTORS,
    FIELD_COLLECTOR_CLASSES, EMBEDDABLE_COLLECTORS), architecture review candidate
    "collapse get_data_status()'s near-identical loops".

    Those three loops previously hand-duplicated this try/construct/call/log/serialize
    body, varying only in constructor arity (config-only / config+store / config_path)
    and the runs_per_day rule -- both kept explicit as construct/cadence_of closures at
    each call site rather than hidden, since they ARE genuinely different per registry
    (not the same logic duplicated). cadence_of returns (runs_per_day,
    runs_per_day_section) as ONE pair from one closure -- not two separate closures
    that could disagree -- since a row's cadence *value* and which section a change to
    it actually saves to are the same underlying fact (see _serialize()'s docstring,
    architecture review candidate "one source of truth for which rows get the cadence
    widget").

    display_label (a class's own override for the Data Status label, e.g.
    GfsAtmosCollector.display_label = "GFS Atmos") is resolved here the same way
    channel_key already is -- a plain getattr with no default, since most collectors
    don't need one and _serialize() falls back to the generic derivation."""
    rows = []
    for cls in classes:
        try:
            channel_key = getattr(cls, "channel_key", None)
            display_label = getattr(cls, "display_label", None)
            instance = construct(cls)
            status = instance.data_status()
            runs_per_day, runs_per_day_section = cadence_of(cls)
            rows.append(
                _serialize(
                    status,
                    channel_key,
                    channel_enabled,
                    runs_per_day,
                    _source_url(instance),
                    runs_per_day_section,
                    display_label,
                )
            )
        except Exception as e:
            logger.error(f"data_status failed for {cls.__name__}: {e}")
    return rows


# --- Class-registry providers (architecture review candidate "Give routers the seam
# the Fakes are waiting for" -- the deferred status.py follow-on). This route's real
# untestability was never the one FieldCatalogAdapter it constructs (fieldstore.get_store
# freezes as a process-wide singleton on first use, so overriding that adapter here
# wouldn't reliably reach it in a test) -- it's the 23 real collector/task classes across
# 5 registries. Injecting the registries lets a test substitute a handful of stub classes
# and exercise this route's OWN logic (iteration, per-class exception swallowing,
# _serialize(), the final envelope) without touching a real DB, config, or the singleton.


def get_collector_classes():
    return COLLECTORS


def get_cache_collector_classes():
    return CACHE_COLLECTORS


def get_field_collector_classes():
    return FIELD_COLLECTOR_CLASSES


def get_embeddable_collector_classes():
    """Pre-resolves EMBEDDABLE_COLLECTORS into a plain list of classes, skipping any whose
    optional dependency isn't installed -- same per-name tolerance resolve_embeddable
    already gives, just resolved once here instead of inline in the route's loop."""
    return [
        cls
        for cls in (resolve_embeddable(name) for name in EMBEDDABLE_COLLECTORS)
        if cls is not None
    ]


def get_task_classes():
    return TASK_CLASSES


def get_process_status_adapter() -> ProcessStatusAdapter:
    return ProcessStatusAdapter()


@router.get("/data_status")
def get_data_status(
    collector_classes=Depends(get_collector_classes),
    cache_collector_classes=Depends(get_cache_collector_classes),
    field_collector_classes=Depends(get_field_collector_classes),
    embeddable_collector_classes=Depends(get_embeddable_collector_classes),
    task_classes=Depends(get_task_classes),
    process_status_adapter=Depends(get_process_status_adapter),
    admin: dict = Depends(require_admin),
):
    try:
        config = load_config()
        workdir = config.get_setting("common", "workdir", ".")
        store = fieldstore.get_store(workdir, field_catalog_adapter=FieldCatalogAdapter())
        channel_enabled = config.get_setting("data_collector", "channel_enabled", {}) or {}

        # Falls back to 1 (CollectorBase.period_s's own default), not None -- a section
        # in RUNS_PER_DAY_SECTIONS always gets the widget, showing whatever value is
        # actually in effect rather than hiding when the key happens to be absent from
        # config. runs_per_day_section is just the row's own section here -- a save
        # always targets the same section the row is about.
        def _event_feed_cadence(cls):
            section = getattr(cls, "section", None)
            # greenhouse_gases' two collectors (ghg_forecast, ghg_egg4_baseline) share
            # one settings section but independent `section` identities (see
            # CollectorBase.settings_section) -- same "shared cadence, one row shows
            # it" shape as gfs_atmos below, surfaced only on the always-required
            # forecast row (matching build_layer_channel_keys()'s own "first collector
            # wins" choice for this pair) rather than duplicating it on both.
            if section == "ghg_forecast":
                return config.get_setting("greenhouse_gases", "runs_per_day", 1), "greenhouse_gases"
            if section not in RUNS_PER_DAY_SECTIONS:
                return None, None
            return config.get_setting(section, "runs_per_day", 1), section

        # data_collector's runs_per_day governs gfs_atmos + gfs_waves + rtofs_currents
        # together; surfaced only on the gfs_atmos row (see RUNS_PER_DAY_SECTIONS
        # docstring above), and a save targets "data_collector", not "gfs_atmos" --
        # returning both facts from this one closure (see cadence_of's docstring) is
        # what lets the frontend stop re-deriving that special case itself. Falls back
        # to 96 (matching CollectorService.refresh_settings's own default), not None.
        #
        # flood_risk_live is the other special case here: FieldCollectorDriver has no
        # is_stale() cadence check of its own (see FloodRiskLiveCollector.collect()'s
        # own comment), so flood_risk.runs_per_day was configured but had no
        # operator-facing control anywhere until now. Surfaced only on this row, not
        # flood_risk_historical -- that row already respects the same shared
        # "flood_risk" section via its own EventFeedDriver-based is_stale() check, and
        # isn't in RUNS_PER_DAY_SECTIONS, so it deliberately shows no duplicate widget.
        def _field_collector_cadence(cls):
            status_name = getattr(cls, "status_name", None)
            if status_name == "gfs_atmos":
                return config.get_setting("data_collector", "runs_per_day", 96), "data_collector"
            if status_name == "flood_risk_live":
                return config.get_setting("flood_risk", "runs_per_day", 24), "flood_risk"
            return None, None

        collectors = _collect_status_rows(
            (*collector_classes, *cache_collector_classes),
            channel_enabled,
            construct=lambda cls: cls(config),
            cadence_of=_event_feed_cadence,
        )
        collectors += _collect_status_rows(
            field_collector_classes,
            channel_enabled,
            construct=lambda cls: cls(config, store),
            cadence_of=_field_collector_cadence,
        )
        collectors += _collect_status_rows(
            embeddable_collector_classes,
            channel_enabled,
            construct=lambda cls: cls(config.config_path),
        )

        layer_channel_keys = build_layer_channel_keys(
            field_collector_classes, cache_collector_classes
        )
        layers = []
        map_data = MapData(config)
        for section, TaskCls in task_classes.items():
            try:
                layers.append(
                    _serialize(
                        TaskCls(config, map_data).layer_status(),
                        layer_channel_keys.get(section),
                        channel_enabled,
                    )
                )
            except Exception as e:
                logger.error(f"layer_status failed for {section}: {e}")

        try:
            layers.append(
                _flightradar_hotspot_status(config, channel_enabled, process_status_adapter)
            )
        except Exception as e:
            logger.error(f"flightradar hotspot status failed: {e}")

        # Group the background-service collectors (satellites_collector,
        # shipping_collector, lightning_collector -- named for the service they run,
        # not the data they gather) after every real data-source row, rather than
        # interleaved with them in whatever order the registries happen to define.
        # Stable sort preserves each group's existing relative order.
        collectors.sort(key=lambda c: c["name"].endswith("_collector"))

        return {"status": "success", "data": {"collectors": collectors, "layers": layers}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data_status/channel_enabled/{channel_key}")
def set_channel_enabled(channel_key: str, payload: dict, admin: dict = Depends(require_admin)):
    """Flip one data_collector.channel_enabled[channel_key] and save immediately --
    deliberately its own tiny endpoint rather than routing through /api/config's POST
    (which replaces the ENTIRE config from the client's masterConfigCache, so a
    partial payload would wipe every other section). The Data Status tab is read-only
    and independent of that big form/masterConfigCache by design (see config.html);
    opting a channel in/out is an operational action that should take effect the
    instant it's clicked, not wait for a separate batch "Save"."""
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=422, detail="enabled must be a boolean")

    config = load_config()
    config.config.setdefault("data_collector", {}).setdefault("channel_enabled", {})
    config.config["data_collector"]["channel_enabled"][channel_key] = enabled
    config.save()
    return {"status": "success", "channel_key": channel_key, "enabled": enabled}


@router.post("/data_status/runs_per_day/{section}")
def set_runs_per_day(section: str, payload: dict, admin: dict = Depends(require_admin)):
    """Set one section's runs_per_day and save immediately -- same rationale as
    set_channel_enabled above: an operational cadence change should take effect the
    instant it's picked, independent of /api/config's whole-config replace. `section`
    is "data_collector" for the GFS/RTOFS field collectors (see RUNS_PER_DAY_SECTIONS'
    docstring) and the section name itself for every other row."""
    runs_per_day = payload.get("runs_per_day")
    if isinstance(runs_per_day, bool) or runs_per_day not in RUNS_PER_DAY_CHOICES:
        raise HTTPException(
            status_code=422,
            detail=f"runs_per_day must be one of {sorted(RUNS_PER_DAY_CHOICES)}",
        )

    config = load_config()
    config.config.setdefault(section, {})["runs_per_day"] = runs_per_day
    config.save()
    return {"status": "success", "section": section, "runs_per_day": runs_per_day}
