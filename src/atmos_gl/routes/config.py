#!/usr/bin/env python3
import json
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.templating import Jinja2Templates
from atmos_gl.db.field_catalog_adapter import FieldCatalogAdapter
from atmos_gl.db.user_adapter import UserAdapter
from atmos_gl.db.user_settings_adapter import UserSettingsAdapter
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.data_status import resolve_run_epoch_utc
from atmos_gl.lib.output_files import OUTFILES
from atmos_gl.routes.auth import current_user_optional, get_user_adapter, require_admin
from atmos_gl.routes.field_specs import (
    FIELD_SPECS,
    field_label,
    section_label,
    format_slider_badge,
    clamp_slider_value,
    to_display_value,
    initial_color_render,
    is_long_or_url_field,
    is_api_key_field,
    validate_against_specs,
)
from datetime import timedelta

router = APIRouter(prefix="/api", tags=["System Configuration"])

# Serves the schema-driven config page directly (see the architecture review's "htmx
# for the configuration UI" candidate) -- no /api prefix, since it returns HTML, not JSON.
# The legacy static ui/config/index.html is retired in favour of this route.
ui_router = APIRouter(tags=["Config UI"])

templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
templates.env.globals["field_specs"] = FIELD_SPECS
templates.env.globals["field_label"] = field_label
templates.env.globals["section_label"] = section_label
templates.env.globals["format_slider_badge"] = format_slider_badge
templates.env.globals["clamp_slider_value"] = clamp_slider_value
templates.env.globals["to_display_value"] = to_display_value
templates.env.globals["initial_color_render"] = initial_color_render
templates.env.globals["is_long_or_url_field"] = is_long_or_url_field
templates.env.globals["is_api_key_field"] = is_api_key_field

# Forecast SOURCES. Each source provides an independent hourly data set with its own
# model run cadence; the frontend treats them uniformly ("give me source X's hours +
# valid times"). A product belongs to exactly one source. The `primary` source drives
# the master scrubber timeline; layers on any other source reconcile their own nearest
# hour by wall-clock valid_time against that master. Adding a new source = one entry
# here (+ its collector handler), not new special-cases.
#
# This replaces the old GFS-oriented-with-currents-exception model: GFS is simply the
# source that happens to be primary, and RTOFS (currents) is just another source.
SOURCES = {
    "gfs": {
        "primary": True,  # drives the master timeline
        "products": [
            "isobars",
            "precipitation",
            "wind",
            "temperature",
            "ozone",
            "stormwatch",
            "waves",
        ],
    },
    "rtofs": {
        "primary": False,
        "products": ["currents"],
    },
}

# Backwards-compatible alias: the GFS-source products (some call sites referenced this).
SCRUBBER_PRODUCTS = SOURCES["gfs"]["products"]


def load_config():
    config_path = os.getenv("CONFIG_PATH", "./config/atmos-gl.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Configuration layout unavailable.")
    config = AtmosGLConfig(config_path)
    config.load()
    return config


def _load_defaults_config() -> dict:
    """Parses config/atmos-gl.json.tmpl -- the tracked template's default values --
    independent of the live config. Backs GET /config/section_defaults/{section} (the
    "Set to Defaults" button); the template path is derived from CONFIG_PATH the same
    way load_config() resolves the live one, since the two always live side by side
    (see CLAUDE.md's Settings changes workflow)."""
    tmpl_path = os.getenv("CONFIG_PATH", "./config/atmos-gl.json") + ".tmpl"
    if not os.path.exists(tmpl_path):
        raise HTTPException(status_code=404, detail="Default configuration template unavailable.")
    with open(tmpl_path) as f:
        return json.load(f)


def get_field_catalog_adapter() -> FieldCatalogAdapter:
    return FieldCatalogAdapter()


def get_user_settings_adapter() -> UserSettingsAdapter:
    return UserSettingsAdapter()


@router.get("/forecast_state")
def get_forecast_state(
    field_catalog_adapter: FieldCatalogAdapter = Depends(get_field_catalog_adapter),
):
    """Run epoch + available forecast hours for the scrubber.

    Returns:
      {
        "status": "success",
        "data": {
          "run_date": "20260613",
          "run_id": "18",
          "run_epoch_utc": "2026-06-13T18:00:00Z",   # valid time of f000
          "fmin": 0, "fmax": 23,
          "hours": [0,1,...,23],
          "max_hour": 23,                             # convenience = fmax
          "valid_times_utc": { "0": "...Z", "1": "...Z", ... }  # per-hour valid time
        }
      }
    """
    try:
        def z(dt):
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        def source_block(products):
            """Build one source's timeline block from whichever of its products actually
            have catalogued DATA (not which layers are toggled on — display state must
            not make the timeline vanish). Intersects hours over the data-present products
            within that source's own freshest run, so model cycles never mix. Returns None
            if the source has no data yet."""
            present = field_catalog_adapter.products_with_data(products)
            if not present:
                return None
            summary = field_catalog_adapter.get_latest_run_hours(products=present)
            if not summary or not summary.get("hours"):
                return None
            epoch = resolve_run_epoch_utc(summary["run_date"], summary["run_id"])
            rdate = summary["run_date"]
            return {
                "run_date": rdate
                if isinstance(rdate, str)
                else rdate.strftime("%Y%m%d"),
                "run_id": summary["run_id"],
                "run_epoch_utc": z(epoch),
                "fmin": summary["fmin"],
                "fmax": summary["fmax"],
                "max_hour": summary["fmax"],
                "hours": summary["hours"],
                "valid_times_utc": {
                    str(h): z(epoch + timedelta(hours=int(h))) for h in summary["hours"]
                },
            }

        # Build every source's block uniformly. `primary` names the source that drives
        # the master scrubber; non-primary sources are reconciled against it by the
        # frontend. The whole timeline is null only if even the primary has no data.
        sources = {}
        primary_name = None
        for name, spec in SOURCES.items():
            block = source_block(spec["products"])
            if block is not None:
                sources[name] = block
            if spec.get("primary"):
                primary_name = name

        if not primary_name or primary_name not in sources:
            # Primary source has no data yet -> no master timeline. (A non-primary source
            # alone can't drive the scrubber.)
            return {"status": "success", "data": None}

        return {
            "status": "success",
            "data": {
                "sources": sources,
                "primary": primary_name,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_config_data() -> dict:
    """Load atmos-gl.json and layer in the frontend RULE__ directives (missing-API-key
    warnings, the shipping stub). Shared by the JSON /api/config endpoint and the
    server-rendered /config page so both see identical data."""
    config = load_config()
    data = config.config.copy()

    # Ensure a frontend directive block exists for the shipping UI module
    if "shipping" not in data:
        data["shipping"] = {"enabled": True}

    ais_key = os.getenv("AIS_API_KEY", "").strip()
    owm_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    maptiler_key = os.getenv("MAPTILER_API_KEY", "").strip()
    firms_key = os.getenv("FIRMS_API_KEY", "").strip()
    cdsapi_key = os.getenv("CDSAPI_KEY", "").strip()
    earthdata_token = os.getenv("EARTHDATA_TOKEN", "").strip()

    if "shipping_collector" in data:
        if not ais_key:
            data["shipping_collector"]["enabled"] = False
            data["shipping_collector"]["RULE__missing_ais"] = True

    if "lightning_collector" in data:
        if not owm_key:
            data["lightning_collector"]["enabled"] = False
            data["lightning_collector"]["RULE__missing_openweather_apikey"] = True

    if "common" in data:
        if not maptiler_key:
            data["common"]["RULE__missing_maptiler"] = True

    if "fires" in data:
        if not firms_key:
            data["fires"]["enabled"] = False
            data["fires"]["RULE__missing_firms_apikey"] = True

    if "greenhouse_gases" in data:
        # Both Absolute and Anomaly modes are sourced from CAMS via the CDS API, so
        # (unlike the original design, where GEOS-CF supplied a keyless Absolute
        # mode) a missing CDSAPI_KEY disables the whole layer -- same shape as every
        # other RULE__missing_* case above. GEOS-CF was dropped after live testing
        # found it doesn't serve CO2 at all; see the published spec's issue comments.
        if not cdsapi_key:
            data["greenhouse_gases"]["enabled"] = False
            data["greenhouse_gases"]["RULE__missing_cdsapi_key"] = True

    if "air_quality" in data:
        # Same CDS/ADS source family as greenhouse_gases -- see that block's comment.
        if not cdsapi_key:
            data["air_quality"]["enabled"] = False
            data["air_quality"]["RULE__missing_cdsapi_key"] = True

    if "flood_risk" in data:
        # Only Live mode (NASA LANCE MODIS observed flooding) needs EARTHDATA_TOKEN --
        # Historical mode (JRC hazard maps, static/no-auth) needs no credential at
        # all, so this gate is mode-specific rather than disabling the whole section
        # the way the single-source greenhouse_gases/air_quality gates above do. See
        # issue #371 and its follow-up grilling (collectors/flood_risk.py's module
        # docstring) for Live mode's data-source pivot away from GloFAS.
        if data["flood_risk"].get("mode") == "live" and not earthdata_token:
            data["flood_risk"]["enabled"] = False
            data["flood_risk"]["RULE__missing_earthdata_token"] = True

    # Not stored in config.json, not user-editable (see lib/output_files.py) -- injected
    # here so the frontend can still read cfg.outfile exactly as before, just sourced
    # from the same hardcoded value the render task itself uses.
    for section, path in OUTFILES.items():
        data.setdefault(section, {})["outfile"] = path

    return data


def _strip_backend_only_secrets(data: dict) -> dict:
    """Drops "api_key" from every section except "common" -- _inject_secrets()
    (lib/config.py) stamps AIS_API_KEY/OPENWEATHER_API_KEY/FIRMS_API_KEY into
    shipping_collector/lightning_collector/fires so the collectors that read
    self.settings["api_key"] at runtime keep working, but those are backend-only
    credentials that must never reach GET /api/config's public, unauthenticated
    response (issue #304 code review). common.api_key (MAPTILER_API_KEY) is the one
    genuine exception: the map itself embeds it directly in client-side MapTiler tile
    requests, so it's already public by design. Builds fresh per-section dicts rather
    than mutating `data` in place -- `data`'s nested dicts are the same objects as the
    live AtmosGLConfig.config's (see _build_config_data()'s shallow .copy()), and
    config_page() (the admin-gated /config page) reuses that same data unfiltered."""
    return {
        section: (
            {k: v for k, v in settings.items() if k != "api_key"}
            if section != "common" and isinstance(settings, dict)
            else settings
        )
        for section, settings in data.items()
    }


def _merge_personal_overrides(data: dict, overrides: dict) -> dict:
    """Applies a signed-in user's sparse {section: {option: value}} overrides on top
    of `data` -- issue #305/#314. Only touches a key when it's both present in
    `overrides` AND currently flagged personalizable=True, so a stale override for a
    key that's since been un-flagged (or removed from FIELD_SPECS entirely) can never
    leak into the response. Builds fresh per-section dicts rather than mutating `data`
    in place, matching _strip_backend_only_secrets' own precaution (data's nested
    dicts are the same objects as the live AtmosGLConfig.config's)."""
    if not overrides:
        return data
    merged = dict(data)
    for section, section_overrides in overrides.items():
        if section not in merged or not isinstance(merged[section], dict):
            continue
        personalizable_values = {
            option: value
            for option, value in section_overrides.items()
            if getattr(FIELD_SPECS.get((section, option)), "personalizable", False)
        }
        if personalizable_values:
            merged[section] = {**merged[section], **personalizable_values}
    return merged


@router.get("/config")
def get_config(
    request: Request,
    user_adapter: UserAdapter = Depends(get_user_adapter),
    settings_adapter: UserSettingsAdapter = Depends(get_user_settings_adapter),
):
    data = _strip_backend_only_secrets(_build_config_data())
    # Unauthenticated by design (see require_admin's docstring) -- current_user_optional
    # called directly rather than as a Depends, so an anonymous request is byte-for-byte
    # identical to before this merge existed, not just "no override values happen to
    # apply" -- no session, no adapter round-trip, no behaviour change at all.
    user = current_user_optional(request, user_adapter)
    if user is not None:
        overrides = settings_adapter.get_overrides(user["id"])
        data = _merge_personal_overrides(data, overrides)
    return {"status": "success", "data": data}


@ui_router.get("/config")
def config_page(request: Request, admin: dict = Depends(require_admin)):
    return templates.TemplateResponse(
        request, "config.html", {"config_data": _build_config_data()}
    )


@ui_router.get("/config/section_defaults/{section}")
def section_defaults(section: str, request: Request, admin: dict = Depends(require_admin)):
    """Renders one section's field grid sourced from config/atmos-gl.json.tmpl's
    values instead of the live config -- backs the config page's "Set to Defaults"
    button. Returns just the fragment (field_macros.render_field_group's output);
    config.html swaps it into #fields-section-{section}'s innerHTML on confirm.
    Reuses render_field_group -- the exact same widget-rendering path config_page's
    own render_tab_group already goes through -- so every field kind (slider/select/
    color/multiselect/grouped_transfer/toggle) gets identical treatment with no
    separate client-side reset logic to keep in sync."""
    section_data = _load_defaults_config().get(section)
    if section_data is None:
        raise HTTPException(status_code=404, detail=f"No defaults found for section '{section}'.")
    return templates.TemplateResponse(
        request, "_section_fields.html", {"section": section, "section_data": section_data}
    )


@router.post("/config")
async def update_config(payload: dict, admin: dict = Depends(require_admin)):
    errors = validate_against_specs(payload)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    config = load_config()

    if "shipping_collector" in payload:
        payload["shipping_collector"].pop("RULE__missing_ais", None)
    if "lightning_collector" in payload:
        payload["lightning_collector"].pop("RULE__missing_openweather_apikey", None)
    if "common" in payload:
        payload["common"].pop("RULE__missing_maptiler", None)
    if "fires" in payload:
        payload["fires"].pop("RULE__missing_firms_apikey", None)
    if "greenhouse_gases" in payload:
        payload["greenhouse_gases"].pop("RULE__missing_cdsapi_key", None)
    if "air_quality" in payload:
        payload["air_quality"].pop("RULE__missing_cdsapi_key", None)
    if "flood_risk" in payload:
        payload["flood_risk"].pop("RULE__missing_earthdata_token", None)

    # outfile is injected read-time-only by _build_config_data() (see OUTFILES/
    # lib/output_files.py) -- never a real stored setting. Strip it the same way the
    # RULE__ flags above are, so a save doesn't persist a client-echoed copy to disk.
    for section in OUTFILES:
        if section in payload:
            payload[section].pop("outfile", None)

    config.config = payload
    config.save()
    return {"status": "success", "message": "Configuration updated successfully."}
