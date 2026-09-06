#!/usr/bin/env python3
"""Declarative widget specs for the schema-driven config UI (architecture review
candidate "htmx for the configuration UI").

Each FIELD_SPECS entry keys (section, option) to the widget that renders and
validates it, replacing the option-name string-matching dispatch in the legacy
client-side config JS (ui/config/index.html's ~46-branch renderTabContainers). A
field with no entry falls back to the existing generic text/number widget -- both
for genuinely generic options and, during the tab-by-tab migration, for any option
not yet ported from the legacy JS. An unspecced boolean value still renders as a
toggle (not the number/text fallback) since every boolean in this config uses the
same widget regardless of field name -- matching the legacy JS's very first dispatch
check, `typeof value === "boolean"`, ahead of any option-name matching.

Many legacy branches matched on option name alone (e.g. any "*_hours" field, any
"*fontsize" field), independent of section -- a handful of module-level spec
constants below capture those shapes once and get registered under every
(section, option) pair that uses them, so the shape is defined a single time.

Migrated so far: Global (common, animation), Events (quakes, volcanoes),
Misc (satellites, terminator, markers), Shipping (shipping),
Weather (clouds, isobars, wind, jetstream, precipitation, pwat, lightning, storms, waves),
Climate (sst, currents, temperature, ozone, stormwatch).

Validated with ast.parse.
"""
from dataclasses import dataclass, field

from atmos_gl.lib.greenhouse_gases import BASELINE_YEAR_MAX, BASELINE_YEAR_MIN


@dataclass(frozen=True)
class ToggleSpec:
    # Issue #305/#314: whether a signed-in user may personally override this setting
    # (persisted per-account, merged on top of the site's global default). Opt-in,
    # defaulting False, curated field-spec by field-spec -- never a setting that
    # changes backend rendering cost/resource use (collector-only settings are never
    # frontend-exposed at all, so they're never even a candidate). Added individually
    # to each of the 6 FieldSpec dataclasses rather than via a shared base class, since
    # this is the first cross-cutting flag any of them has needed.
    personalizable: bool = False
    kind: str = field(default="toggle", init=False)


@dataclass(frozen=True)
class SliderSpec:
    min: float
    max: float
    step: float
    # Badge display: None = show the raw value verbatim (matches fields whose
    # legacy JS badge did no reformatting); an int fixes the decimal places shown.
    decimals: int | None = None
    prefix: str = ""
    suffix: str = ""
    # value == 0 renders as this instead of the normal number+suffix (e.g. "off",
    # "keep forever") -- ported from legacy fields with a sentinel-value badge.
    zero_label: str | None = None
    # Appends "s" to `suffix` when the value isn't exactly 1 (e.g. "1 day" / "5 days").
    pluralize: bool = False
    # True only for clouds.threshold: the stored/posted value is a raw 0-255 byte,
    # but the slider displays/edits it as a 0-100 percentage. `min`/`max`/`step`
    # describe the DISPLAYED (percent) slider; `raw_max` is the stored value's actual
    # max, used by to_display_value (render) and validate_against_specs (POST).
    byte_to_percent: bool = False
    raw_max: float | None = None
    # CSS hook for the legacy saveActiveConfig() JS's per-class save dispatch (e.g.
    # "cloud-threshold-slider" triggers its percent -> byte reverse conversion).
    extra_class: str = ""
    personalizable: bool = False
    kind: str = field(default="slider", init=False)


@dataclass(frozen=True)
class SelectSpec:
    options: list  # [(value, label), ...]
    personalizable: bool = False
    kind: str = field(default="select", init=False)


@dataclass(frozen=True)
class MultiSelectSpec:
    options: list  # [(value, label), ...]
    personalizable: bool = False
    kind: str = field(default="multiselect", init=False)


@dataclass(frozen=True)
class GroupedTransferSpec:
    """Two-listbox 'shuttle' control: groups is [(heading, [(value, label), ...]), ...].
    Inactive options render in the left listbox, active (currently selected) in the
    right, both grouped/indented under the same headings via <optgroup> -- Add/Remove
    buttons move selected options between the two, keeping each option under its own
    heading either side."""
    groups: list
    personalizable: bool = False
    kind: str = field(default="grouped_transfer", init=False)


@dataclass(frozen=True)
class ColorSpec:
    # True (the common case): saved as the nearest named colour (e.g. "White"),
    # like markers.marker_color / volcanoes.marker_color. False: saved as the raw
    # hex string, like terminator.shade_color -- ported from the legacy JS's
    # pickerClass distinction (option.includes('_default_') or section == 'terminator'
    # got the raw-hex behaviour; everything else got the named-colour behaviour).
    named: bool = True
    personalizable: bool = False
    kind: str = field(default="color", init=False)


# Mirrors the client-side COLOR_MAP in templates/config.html -- needed server-side
# only to resolve a stored *name* (e.g. "white") to its initial hex swatch value;
# the reverse direction (hex -> nearest name) stays client-side in findNearestNamedColor,
# used live as the user drags the picker.
COLOR_MAP = {
    "white": "#ffffff", "black": "#000000", "gray": "#808080", "silver": "#c0c0c0",
    "red": "#ff0000", "maroon": "#800000", "pink": "#ffc0cb",
    "green": "#00ff00", "lime": "#00ff00", "olive": "#808000", "teal": "#008080",
    "blue": "#0000ff", "navy": "#000080", "cyan": "#00ffff", "aqua": "#00ffff",
    "yellow": "#ffff00", "orange": "#ffa500", "gold": "#ffd700",
    "purple": "#800080", "magenta": "#ff00ff", "violet": "#ee82ee",
}


def initial_color_render(value) -> tuple[str, str]:
    """(hex, label) for a color field's initial render -- ported verbatim from the
    legacy JS (which capitalizes the raw stored string rather than computing the
    nearest named color; that computation only happens client-side, on interaction)."""
    raw = str(value).lower().strip()
    hex_value = raw if raw.startswith("#") else COLOR_MAP.get(raw, "#ffffff")
    label = (raw[:1].upper() + raw[1:]) if raw else "White"
    return hex_value, label


# --- Shared shapes: legacy branches matched these purely on option name,
# regardless of section, so one instance is registered under every field that
# uses it (see FIELD_SPECS below). ---

# Every one of its 5 call sites (quakes, volcanoes, flightradar, shipping, lightning)
# is a personalizable layer's own marker-icon scale -- unlike _OPACITY/_ICON_ZOOM's
# sibling constants below, none of its usages are baked server-side, so the shared
# instance itself is safely personalizable rather than needing per-site overrides.
_ICON_ZOOM = SliderSpec(min=0.1, max=5.0, step=0.1, decimals=1, suffix="x", personalizable=True)
_HOURS = SliderSpec(min=0, max=96, step=1, suffix="h")
_MINUTES = SliderSpec(min=0, max=120, step=1, suffix="mins", personalizable=True)
_FONTSIZE = SliderSpec(min=6, max=24, step=1, suffix="px", personalizable=True)
# Mixed personalizability: volcanoes.smoke_opacity and air_quality.opacity are baked
# server-side into a shared render (so2_volcanic's/every air_quality variable's alpha
# feathering -- see tasks/air_quality.py's plot()) and isobars.opacity currently has no
# live effect on the WebGL fill-mode render at all (only isobars.py's own separate
# static-PNG label alpha, out of #315's scope) -- those three sites get their own
# dedicated, non-personalizable SliderSpec instead of this shared one. See issue #315.
_OPACITY = SliderSpec(min=0, max=100, step=1, personalizable=True)
_PARTICLE_OPACITY = SliderSpec(min=0, max=100, step=5, personalizable=True)
_PARTICLE_SPEED_LIKE = SliderSpec(min=0, max=100, step=1, personalizable=True)
_PARTICLE_SIZE = SliderSpec(min=0.1, max=5.0, step=0.05, decimals=2, personalizable=True)
_TRAIL_LENGTH = SliderSpec(min=0, max=100, step=1, personalizable=True)
# Streamline-ribbon half-thickness (_currentparticles_gl.js's curThick, shared by wind
# and currents -- the unified engine both now render through).
_TRAIL_THICKNESS = SliderSpec(min=0.5, max=5.0, step=0.1, decimals=1, suffix="px", personalizable=True)
# Direction-coherence smoothing radius (_currentparticles_gl.js's coherenceRadius) --
# needed by any consumer reading a raw 0.25deg GFS field (wind, jetstream), whose
# small-scale grid noise otherwise reads as trails jittering between paths frame to
# frame. Currents never sets this (RTOFS is smooth enough not to need it).
_FLOW_COHERENCE_RADIUS = SliderSpec(min=0.0, max=10.0, step=0.5, decimals=2, personalizable=True)
_MIN_MAX_C = SliderSpec(min=0, max=36, step=1, suffix=" DegC", personalizable=True)
# Every layer section's Show-tab visibility flag (issue #313) -- registered so
# `enabled` is a real, validated FIELD_SPECS entry like every other setting
# (closing a prior gap where any value type was silently accepted), even though
# render_field_group's own "enabled" exclusion (see _field_macros.html) keeps it
# out of the generic properties-tab rendering for every section except
# housekeeper -- the Show tab's own hardcoded checkboxes/radios remain the
# actual UI for it, unchanged by this promotion.
_ENABLED = ToggleSpec()
# Same shape, personalizable (issue #315) -- layer visibility never affects backend
# rendering cost (the backend renders unconditionally of any frontend enabled flag,
# see Updater.layer_status()'s docstring), so every layer EXCEPT the backend
# collector/housekeeper sections (which have no visible "layer" of their own -- their
# enabled flag gates a data-acquisition loop, not something shown on the map) uses
# this personalizable variant instead of _ENABLED above.
_ENABLED_PERSONALIZABLE = ToggleSpec(personalizable=True)
_CACHE_EXPIRY_DAYS = SliderSpec(
    min=0, max=30, step=1, suffix=" day", zero_label="keep forever", pluralize=True
)
# Shared by shipping_collector/lightning_collector's "Sleep interval" -- the pause
# between collection passes for each long-running async collector. Stored/edited in
# minutes (this slider); each collector converts to seconds itself (see
# _sleep_interval_seconds() on both classes).
_SLEEP_INTERVAL_MINUTES = SliderSpec(min=5, max=30, step=1, suffix=" min", pluralize=True)
# shipping_collector.vessel_track_expiry_days: 0 already means "never prune" in
# ShipAdapter.prune_vessel_tracks() (an `if not expiry_days or expiry_days <= 0: return`
# guard predating this slider) -- zero_label makes the UI say so instead of "0 days".
_VESSEL_TRACK_EXPIRY_DAYS = SliderSpec(
    min=0, max=60, step=5, suffix=" day", zero_label="Never", pluralize=True
)
# shipping_collector.listen_duration -- base per-slice AIS listen time, stored/edited in
# minutes (this slider); ShippingCollector converts to seconds itself (see
# _listen_duration_seconds()).
_LISTEN_DURATION_MINUTES = SliderSpec(min=5, max=60, step=5, suffix=" min", pluralize=True)

_LEVEL_OF_DETAIL = SelectSpec([
    ("1", "Low resolution"),
    ("2", "Medium resolution"),
    ("3", "High resolution (needs lots of memory)"),
])

# common.performance_tier -- caps LayerBuilder's concurrent render-worker count (see
# workers_for_tier() in layer_builder.py). Deliberately lives in `common`, not
# `layer_builder`'s own section, as a general "how much CPU/RAM is this host willing to
# spend" switch -- see docs/adr/0006-performance-tier-lives-in-common.md.
_PERFORMANCE_TIER = SelectSpec([
    ("low", "Low (safest — minimal memory/CPU use, slower rendering)"),
    ("medium", "Medium (balanced, default)"),
    ("high", "High (fastest rendering, needs more memory/CPU)"),
])

_MODE_OPTIONS = SelectSpec([
    ("absolute", "Absolute"),
    ("anomaly", "Anomaly"),
], personalizable=True)

_GHG_SPECIES = SelectSpec([
    ("co2", "CO2"),
    ("ch4", "CH4 (Methane)"),
], personalizable=True)

# Flood Risk's two modes are independently-sourced metrics, not two views of the same
# data (see issue #371's design grill and its follow-up) -- Live is NASA LANCE MODIS
# observed flooding (needs EARTHDATA_TOKEN; originally a daily GloFAS forecast,
# abandoned for unfixable OOM/network problems -- see collectors/flood_risk.py's
# module docstring), Historical is a fixed JRC hazard classification (no credential).
# Personalizable like every other client-side-only mode toggle (post-#312): switching
# it just changes which pre-rendered texture this layer's next poll tick fetches.
_FLOOD_RISK_MODE = SelectSpec([
    ("live", "Live (Observed Inundation)"),
    ("historical", "Historical (JRC 100yr Hazard)"),
], personalizable=True)

# CAMS EGG4 reanalysis (the anomaly baseline source) was never extended past 2020, so
# the picker only offers years it actually has gridded data for -- see
# lib/greenhouse_gases.BASELINE_YEAR_MIN/MAX and the GHG design grill.
_GHG_BASELINE_YEAR = SelectSpec(
    [(str(y), str(y)) for y in range(BASELINE_YEAR_MIN, BASELINE_YEAR_MAX + 1)]
)

_GHG_PALETTE = SelectSpec([
    ("thermal", "Thermal"),
    ("vivid", "Vivid"),
    ("deep", "Deep"),
    ("ocean", "Ocean"),
], personalizable=True)

# Thresholds are the US EPA AQI "Unhealthy for Sensitive Groups" 24-hr breakpoints
# (PM2.5 35.5 ug/m3, PM10 155 ug/m3; both rounded here) -- the same AQI framing
# tasks/air_quality.py's AOD-default comment already leans on, and consistent with
# this layer's AQI-style green->purple colour gradient (_AQI_COLORS). AOD has no
# equivalent official breakpoint (see that comment), so its label is left as-is.
_AQ_VARIABLE = SelectSpec([
    ("pm2_5", "PM2.5 (Unhealthy above 35 µg/m³)"),
    ("pm10", "PM10 (Unhealthy above 150 µg/m³)"),
    ("aod", "Smoke (AOD)"),
    ("so2", "SO2 (Sulphur Dioxide)"),
], personalizable=True)

_LOG_LEVEL = SelectSpec([
    ("DEBUG", "DEBUG"),
    ("INFO", "INFO"),
    ("WARNING", "WARNING"),
    ("ERROR", "ERROR"),
    ("CRITICAL", "CRITICAL"),
])

_FIRE_CONFIDENCE = SelectSpec([
    ("low", "Low - include all detections"),
    ("nominal", "Nominal - filter out low-confidence noise"),
    ("high", "High - saturated pixels only"),
], personalizable=True)

# Real wildfire fronts, even the most extreme recorded, top out in the low-thousands of
# MW per pixel -- readings far above that are far more likely a gas flare/industrial
# source than an actual fire (see FireAdapter.get_fires_as_geojson's docstring).
_FIRE_MAX_FRP = SliderSpec(min=500, max=20000, step=100, suffix=" MW", personalizable=True)

_SAT_NAMES = MultiSelectSpec([
    ("ISS (ZARYA)", "ISS (ZARYA) - International Space Station"),
    ("CSS (TIANHE)", "CSS (TIANHE)  - Chinese Space Station"),
    ("HST", "HST - Hubble Space Telescope"),
    ("FGRST (GLAST)", "FGRST (GLAST) - The Fermi Gamma-ray Space Telescope"),
    ("SWIFT", "SWIFT - The Neil Gehrels Swift Observatory"),
    ("NOAA 15", "NOAA 15 - The Polar orbiting weather fleet"),
    ("NOAA 18", "NOAA 18"),
    ("NOAA 19", "NOAA 19"),
    ("NOAA 20", "NOAA 20"),
    ("NOAA 21", "NOAA 21"),
    ("AQUA", "AQUA - NASA flagship water-cycle observer."),
    ("TERRA", "TERRA - Twin to Aqua, tasked with mapping land mass and vegetation"),
    ("LANDSAT 8", "LANDSAT 8 - Legendary optical and thermal Earth-imaging satellite"),
    ("LANDSAT 9", "LANDSAT 9 - The newest Landsat satellite"),
    ("SENTINEL-1A", "SENTINEL-1A - European Space Agency primary radar imaging satellite"),
    ("GOES 16", "GOES 16 - Geostationary (Americas/Atlantic)"),
    ("GOES 18", "GOES 18 - Geostationary (Pacific/Americas)"),
    ("METEOSAT-9", "METEOSAT-9 - Geostationary (Indian Ocean)"),
    ("METEOSAT-10", "METEOSAT-10 - Geostationary (Europe/Africa)"),
    ("METEOSAT-11", "METEOSAT-11 - Geostationary (Prime Meridian)"),
], personalizable=True)

# CelesTrak's own catalog groupings (https://celestrak.org/NORAD/elements/) -- these
# section headings and group slugs are exactly as CelesTrak organizes/names them; the
# slugs are also the valid GROUP= query parameter values satellites.py's _fetch_group()
# passes straight through to CelesTrak's GP data API.
_CELESTRAK_GROUPS = GroupedTransferSpec([
    ("Special-Interest Satellites", [
        ("last-30-days", "Last 30 Days' Launches"),
        ("stations", "Space Stations"),
        ("visual", "100 (or so) Brightest"),
        ("active", "Active Satellites"),
        ("analyst", "Analyst Satellites"),
        ("fengyun-1c-debris", "Fengyun 1C Debris"),
        ("iridium-33-debris", "Iridium 33 Debris"),
        ("cosmos-2251-debris", "Cosmos 2251 Debris"),
    ]),
    ("Weather & Earth Resources Satellites", [
        ("weather", "Weather"),
        ("resource", "Earth Resources"),
        ("sar", "Synthetic Aperture Radar (SAR)"),
        ("sarsat", "Search & Rescue (SARSAT)"),
        ("dmc", "Disaster Monitoring Constellation"),
        ("tdrss", "Tracking & Data Relay Satellite System"),
        ("argos", "ARGOS Data Collection"),
        ("planet", "Planet Labs"),
        ("spire", "Spire Global"),
    ]),
    ("Communications Satellites", [
        ("geo", "Active Geosynchronous"),
        ("gpz", "GPZ"),
        ("gpz-plus", "GPZ Plus"),
        ("intelsat", "Intelsat"),
        ("ses", "SES"),
        ("eutelsat", "Eutelsat"),
        ("telesat", "Telesat"),
        ("starlink", "Starlink"),
        ("oneweb", "OneWeb"),
        ("qianfan", "Qianfan (China Satellite Network)"),
        ("hulianwang", "Hulianwang (Guowang)"),
        ("kuiper", "Amazon Kuiper"),
        ("iridium-NEXT", "Iridium NEXT"),
        ("orbcomm", "Orbcomm"),
        ("globalstar", "Globalstar"),
        ("amateur", "Amateur Radio"),
        ("satnogs", "SatNOGS"),
        ("x-comm", "Experimental Comms"),
        ("other-comm", "Other Comms"),
    ]),
    ("Navigation Satellites", [
        ("gnss", "All GNSS"),
        ("gps-ops", "GPS Operational"),
        ("glo-ops", "GLONASS Operational"),
        ("galileo", "Galileo"),
        ("beidou", "BeiDou"),
        ("sbas", "Satellite-Based Augmentation (SBAS)"),
    ]),
    ("Scientific Satellites", [
        ("science", "Space & Earth Science"),
        ("geodetic", "Geodetic"),
        ("engineering", "Engineering"),
        ("education", "Education"),
    ]),
    ("Miscellaneous Satellites", [
        ("military", "Miscellaneous Military"),
        ("radar", "Radar Calibration"),
        ("cubesat", "CubeSats"),
    ]),
])


FIELD_SPECS = {
    # --- Global (common, animation) ---
    # basemap/atmosphere/auto_rotate*/starting_lat*/key_fontsize are pure client-side
    # viewport/display preferences -- none of them are backend-render-cost-affecting
    # (a basemap swap and camera position never touch the render pipeline), so they're
    # personalizable "camera/viewport default" settings (issue #305/#315). log_level
    # and performance_tier stay admin-only -- both are genuine backend operational
    # controls, not display preferences.
    ("common", "basemap"): SelectSpec([
        ("satellite", "Satellite"),
        ("hybrid", "Satellite + Labels"),
        ("streets-v2", "Streets"),
        ("outdoor-v2", "Outdoor / Terrain"),
        ("topo-v2", "Topographic"),
        ("dataviz-dark", "Dataviz Dark"),
        ("winter", "Winter"),
        ("basic-v2", "Basic"),
    ], personalizable=True),
    ("common", "atmosphere"): ToggleSpec(personalizable=True),
    ("common", "auto_rotate"): ToggleSpec(personalizable=True),
    ("common", "auto_rotate_speed"): SliderSpec(min=0.01, max=1.0, step=0.01, personalizable=True),
    # Fixes a pre-existing bug in the legacy JS, which swapped these two ranges
    # (latitude got +/-180, longitude got +/-90).
    ("common", "starting_latitude"): SliderSpec(
        min=-90.0, max=90.0, step=1.0, decimals=1, suffix=" deg", personalizable=True
    ),
    ("common", "starting_longitude"): SliderSpec(
        min=-180.0, max=180.0, step=1.0, decimals=1, suffix=" deg", personalizable=True
    ),
    ("common", "log_level"): _LOG_LEVEL,
    ("common", "performance_tier"): _PERFORMANCE_TIER,
    # Single shared fontsize for every layer's colourbar-key PNG (sst/waves/currents/
    # ozone/precipitation/temperature/wind/jetstream/stormwatch/pwat), replacing what
    # used to be 11 identical per-layer key_fontsize settings -- consolidated here
    # since every layer already defaulted to the same value (10) and there was no
    # actual per-layer customization to preserve.
    ("common", "key_fontsize"): _FONTSIZE,
    ("animation", "forecast_stepping"): ToggleSpec(personalizable=True),
    ("animation", "stepping_rate"): _PARTICLE_SPEED_LIKE,
    # --- Events (quakes, volcanoes) ---
    ("quakes", "enabled"): _ENABLED_PERSONALIZABLE,
    ("quakes", "icon_zoom"): _ICON_ZOOM,
    # _HOURS itself stays non-personalizable (mixed usage elsewhere -- see its own
    # comment); quakes' two hour sliders are pure client-side marker-recency filters,
    # so they get their own dedicated, personalizable instances.
    ("quakes", "recent_activity_hours"): SliderSpec(min=0, max=96, step=1, suffix="h", personalizable=True),
    ("quakes", "expiry_hours"): SliderSpec(min=0, max=96, step=1, suffix="h", personalizable=True),
    ("quakes", "label_fontsize"): _FONTSIZE,
    ("quakes", "min_mag"): SliderSpec(min=0, max=10, step=0.1, decimals=1, prefix="M ", personalizable=True),
    ("volcanoes", "enabled"): _ENABLED_PERSONALIZABLE,
    ("volcanoes", "icon_zoom"): _ICON_ZOOM,
    # "Show Smoke Plume" (issue #254) -- Volcano Properties is the SOLE owner of the
    # volcanic-specific SO2 variable's opacity/threshold settings (so2_volcanic, NOT
    # the general SO2 air_quality's own picker offers -- the two are genuinely
    # different CDS variables, see tasks/air_quality.py's _CAMS_VARS and
    # lib/air_quality.py's module docstring) -- see tasks/air_quality.py's
    # _SETTINGS_SECTION_OVERRIDE. smoke_opacity/so2_min are deliberately separate keys
    # from any other opacity/threshold concept, since volcanoes has no other opacity
    # setting today. so2_min's range matches ("air_quality", "so2_min") below -- same
    # magnitude confirmed live (tasks/air_quality.py's _DEFAULT_MIN comment) despite
    # being a different physical quantity -- but kept as a genuinely separate slider,
    # not the same one, since the two variables can still legitimately diverge.
    ("volcanoes", "show_smoke_plume"): ToggleSpec(personalizable=True),
    # NOT personalizable, unlike every other opacity/threshold in this app: these two
    # are read server-side by AirQualityUpdater.plot() (via _SETTINGS_SECTION_OVERRIDE)
    # and baked into the shared so2_volcanic render's alpha feathering -- personalizing
    # them would need a per-user server re-render, which this architecture doesn't do.
    # A dedicated instance (not the shared, now-personalizable _OPACITY) for that reason.
    ("volcanoes", "smoke_opacity"): SliderSpec(min=0, max=100, step=1),
    ("volcanoes", "so2_min"): SliderSpec(min=0, max=20, step=0.5, decimals=1, suffix=" DU"),
    ("fires", "enabled"): _ENABLED_PERSONALIZABLE,
    ("fires", "expiry_hours"): SliderSpec(min=0, max=96, step=1, suffix="h", personalizable=True),
    ("fires", "min_confidence"): _FIRE_CONFIDENCE,
    ("fires", "max_frp"): _FIRE_MAX_FRP,
    # Fire Weather Index heatmap (tasks/fire_weather.py) -- same "fires" section as the
    # FIRMS hotspot settings above, so the Show tab needs only one "Wildfires" toggle.
    ("fires", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("fires", "opacity"): _OPACITY,
    ("fires", "min_risk_display"): SliderSpec(min=0, max=100, step=5, suffix="", personalizable=True),
    ("fires", "min_risk_filter"): SliderSpec(
        min=0, max=100, step=5, suffix="", zero_label="off", personalizable=True
    ),
    # World Events (GDELT Event Database 2.0, curated CAMEO code allowlist -- see
    # collectors/world_events.py). expiry_days is a pure read-time WHERE filter
    # (WorldEventAdapter.get_events_as_geojson) -- every viewer shares the same
    # underlying rows, only how far back a given query looks changes, same shape as
    # quakes'/fires' own personalizable expiry sliders. min_mentions/backfill_days
    # are NOT personalizable: unlike fires' confidence/frp (filtered only at read
    # time, every detection is always stored), min_mentions gates what
    # WorldEventsCollector.collect() stores in the first place -- a real,
    # shared collection-cost control, not a per-viewer display preference.
    ("world_events", "enabled"): _ENABLED_PERSONALIZABLE,
    ("world_events", "opacity"): _OPACITY,
    ("world_events", "marker_size"): SliderSpec(
        min=0.5, max=3.0, step=0.1, decimals=1, suffix="x", personalizable=True
    ),
    ("world_events", "expiry_days"): SliderSpec(
        min=1, max=14, step=1, suffix=" day", pluralize=True, personalizable=True
    ),
    ("world_events", "min_mentions"): SliderSpec(min=0, max=100, step=5),
    ("world_events", "backfill_days"): SliderSpec(
        min=1, max=14, step=1, suffix=" day", pluralize=True
    ),
    ("world_events", "show_explosion"): ToggleSpec(personalizable=True),
    ("world_events", "show_warfare"): ToggleSpec(personalizable=True),
    ("world_events", "show_targeted_violence"): ToggleSpec(personalizable=True),
    ("world_events", "show_diplomacy"): ToggleSpec(personalizable=True),
    # Troublespots (issue #366) -- a derived multi-domain convergence layer over
    # World Events/Earthquakes/Fires/Volcanic Activity, computed live per request (no
    # table, no collector of its own). cell_size_deg/window_hours are NOT
    # personalizable, unlike every other layer's opacity/expiry: they control the
    # underlying convergence computation itself, which is meant to be one objective,
    # shared signal every viewer sees the same way -- letting each user tune their own
    # severity map would undermine that (see the design's roster/config decision).
    ("troublespots", "enabled"): _ENABLED_PERSONALIZABLE,
    ("troublespots", "opacity"): _OPACITY,
    ("troublespots", "cell_size_deg"): SliderSpec(
        min=1.0, max=5.0, step=0.5, decimals=1, suffix=" deg"
    ),
    ("troublespots", "window_hours"): SliderSpec(
        min=12, max=168, step=12, suffix="h"
    ),
    # --- Misc (satellites, terminator, markers, flightradar) ---
    ("satellites", "enabled"): _ENABLED_PERSONALIZABLE,
    ("satellites", "sat_names"): _SAT_NAMES,
    ("satellites", "past_minutes"): _MINUTES,
    ("satellites", "future_minutes"): _MINUTES,
    ("satellites", "step_seconds"): SliderSpec(min=5, max=120, step=5, suffix="s", personalizable=True),
    ("satellites", "color"): ColorSpec(personalizable=True),
    ("terminator", "enabled"): _ENABLED_PERSONALIZABLE,
    ("terminator", "opacity"): _OPACITY,
    ("terminator", "shade_color"): ColorSpec(named=False, personalizable=True),
    ("terminator", "edge_softness"): SliderSpec(min=0, max=50, step=1, personalizable=True),
    ("markers", "enabled"): _ENABLED_PERSONALIZABLE,
    ("markers", "marker_color"): ColorSpec(personalizable=True),
    ("markers", "marker_fontsize"): _FONTSIZE,
    ("markers", "weather_popup"): ToggleSpec(personalizable=True),
    ("flightradar", "enabled"): _ENABLED_PERSONALIZABLE,
    ("flightradar", "icon_zoom"): _ICON_ZOOM,
    # Track shown only while hovering an aircraft (flightradar.js) -- same hover-only
    # shape as shipping's track below, not a persistent overlay.
    ("flightradar", "view_tracks"): ToggleSpec(personalizable=True),
    ("flightradar", "track_limit"): SliderSpec(min=5, max=100, step=5, personalizable=True),
    ("flightradar", "track_color"): ColorSpec(personalizable=True),
    # Coastline/lake-shore outline overlay -- a halo'd pair of stroke colours (main +
    # contrasting halo, see ui/modules/landmass.js) rather than a single colour, so it
    # stays legible over any basemap/data-layer combination underneath.
    ("landmass", "enabled"): _ENABLED_PERSONALIZABLE,
    ("landmass", "color"): ColorSpec(personalizable=True),
    ("landmass", "halo_color"): ColorSpec(personalizable=True),
    ("landmass", "linewidth"): SliderSpec(min=0.2, max=5.0, step=0.1, decimals=1, suffix="px", personalizable=True),
    ("landmass", "opacity"): _OPACITY,
    # --- Shipping (shipping) ---
    ("shipping", "enabled"): _ENABLED_PERSONALIZABLE,
    ("shipping", "icon_zoom"): _ICON_ZOOM,
    # Track shown only while hovering a ship (shipping.js) -- not a persistent overlay,
    # so there's no opacity/always-on styling to match here, just the three knobs the
    # hover-track itself needs.
    ("shipping", "view_tracks"): ToggleSpec(personalizable=True),
    ("shipping", "track_limit"): SliderSpec(min=5, max=100, step=5, personalizable=True),
    ("shipping", "track_color"): ColorSpec(personalizable=True),
    # --- Weather (clouds, isobars, wind, jetstream, precipitation, pwat, lightning, storms, waves) ---
    # Only `enabled` is personalizable for clouds: threshold/gamma are baked
    # server-side into the shared transparent overlay (CloudUpdater.
    # save_cache_as_transparent), and offset_days/expiry_hours/cache_expiry_days are
    # collector-cost concerns (which day's GIBS imagery gets fetched, cache retention).
    ("clouds", "enabled"): _ENABLED_PERSONALIZABLE,
    ("clouds", "threshold"): SliderSpec(
        min=0, max=100, step=1, suffix="%",
        byte_to_percent=True, raw_max=255, extra_class="cloud-threshold-slider",
    ),
    ("clouds", "gamma"): SliderSpec(min=0.1, max=3.0, step=0.05, decimals=2, prefix="γ "),
    ("clouds", "offset_days"): SliderSpec(min=0, max=7, step=1, suffix=" days"),
    ("clouds", "expiry_hours"): _HOURS,
    ("clouds", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("isobars", "enabled"): _ENABLED_PERSONALIZABLE,
    ("isobars", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("isobars", "isobar_step"): SliderSpec(min=1, max=10, step=1, suffix=" hPa", personalizable=True),
    ("isobars", "isobar_color"): ColorSpec(personalizable=True),
    ("isobars", "linewidth"): SliderSpec(min=0.1, max=5.0, step=0.1, decimals=1, suffix="px", personalizable=True),
    # NOT personalizable, unlike every other layer's opacity: isobars.js's WebGL fill
    # mode never wires this into its shader (no u_alpha/opacityUniform customUniform --
    # only u_interval/u_linewidth/u_linecolor) -- it currently only affects isobars.py's
    # own separate static-PNG label alpha (out of #315's scope), so it has zero visible
    # effect in the common (WebGL) rendering path. A dedicated instance rather than the
    # shared, now-personalizable _OPACITY -- exposing a control with no live effect
    # would be misleading, not just an edge case.
    ("isobars", "opacity"): SliderSpec(min=0, max=100, step=1),
    ("isobars", "label_fontsize"): _FONTSIZE,
    ("isobars", "label_outline"): ToggleSpec(),
    ("isobars", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    # Ordered to mirror currents' shape below (same shared engine): resolution, colour,
    # opacity, particle tuning, field-quality knobs, trail rendering, playback quality.
    ("wind", "enabled"): _ENABLED_PERSONALIZABLE,
    ("wind", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("wind", "vector_color"): ColorSpec(personalizable=True),
    ("wind", "opacity"): _OPACITY,
    # particle_speed/trail_length/trail_thickness use wind-specific ranges (not the
    # shared _PARTICLE_SPEED_LIKE/_TRAIL_LENGTH/_TRAIL_THICKNESS specs currents also
    # uses) -- live tuning found wind's useful values sitting in a much narrower band,
    # so its sliders were rescaled to give that band the full 0-100 (or 1-5) resolution.
    ("wind", "particle_speed"): SliderSpec(min=10, max=100, step=1, personalizable=True),
    ("wind", "particle_opacity"): _PARTICLE_OPACITY,
    ("wind", "flow_coherence_radius"): _FLOW_COHERENCE_RADIUS,
    ("wind", "trail_length"): SliderSpec(min=10, max=100, step=1, personalizable=True),
    ("wind", "trail_thickness"): SliderSpec(min=1, max=5, step=1, personalizable=True),
    ("wind", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    # Jet stream is speed-colored particles with no heatmap, like currents (not
    # wind's flat-colored particles + separate heatmap) -- shares currents' particle
    # tuning ranges (_PARTICLE_SPEED_LIKE/_TRAIL_LENGTH/_TRAIL_THICKNESS) rather than
    # wind's rescaled ones. Lives on the Weather tab alongside wind, not Climate
    # alongside currents: wind/jetstream/waves are all part of the same GFS forecast
    # cycle (GfsAtmosCollector/GfsWavesCollector, datasource_key "gfs"), unlike
    # currents' separate RTOFS ocean-circulation model. flow_coherence_radius reuses
    # WIND's spec/mechanism though, not currents' (which has none) -- jetstream reads
    # the same noisy 0.25deg GFS grid wind does, unlike currents' smooth RTOFS source.
    ("jetstream", "enabled"): _ENABLED_PERSONALIZABLE,
    ("jetstream", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("jetstream", "palette"): SelectSpec([
        ("stratosphere", "Stratosphere"),
        ("aurora", "Aurora (green -> violet)"),
        ("inferno", "Inferno (orange -> yellow)"),
    ], personalizable=True),
    ("jetstream", "opacity"): _OPACITY,
    ("jetstream", "particle_speed"): _PARTICLE_SPEED_LIKE,
    ("jetstream", "particle_opacity"): _PARTICLE_OPACITY,
    ("jetstream", "flow_coherence_radius"): _FLOW_COHERENCE_RADIUS,
    ("jetstream", "trail_length"): _TRAIL_LENGTH,
    ("jetstream", "trail_thickness"): _TRAIL_THICKNESS,
    ("jetstream", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("precipitation", "enabled"): _ENABLED_PERSONALIZABLE,
    ("precipitation", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("precipitation", "min_mm_hr"): SliderSpec(
        min=0.0, max=10.0, step=0.1, decimals=1, personalizable=True
    ),
    # A dedicated SliderSpec rather than the shared _OPACITY constant (frozen, so it
    # can't be flipped in place) -- deliberately scoped to Precipitation alone for
    # #314, the personalization mechanism's proof-of-concept layer. #315 curates the
    # rest of the app's fields, including every other section's opacity.
    ("precipitation", "opacity"): SliderSpec(min=0, max=100, step=1, personalizable=True),
    ("precipitation", "palette"): SelectSpec([
        ("standard", "Standard"),
        ("ocean_blue", "Ocean blue"),
        ("high_contrast", "High contrast"),
    ], personalizable=True),
    ("precipitation", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("pwat", "enabled"): _ENABLED_PERSONALIZABLE,
    ("pwat", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("pwat", "palette"): SelectSpec([
        ("standard", "Standard (matches precipitation)"),
        ("atmospheric_river", "Atmospheric river (blue -> violet)"),
        ("deep_teal", "Deep teal (cyan -> teal)"),
    ], personalizable=True),
    ("pwat", "critical_pwat"): SliderSpec(min=0.0, max=80.0, step=5.0, decimals=0, suffix="mm", personalizable=True),
    ("pwat", "opacity"): _OPACITY,
    ("pwat", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("lightning", "enabled"): _ENABLED_PERSONALIZABLE,
    ("lightning", "icon_zoom"): _ICON_ZOOM,
    ("lightning", "strike_recent_minutes"): _MINUTES,
    ("lightning", "strike_keep_minutes"): _MINUTES,
    ("lightning", "strike_expiry_hours"): SliderSpec(min=0, max=96, step=1, suffix="h", personalizable=True),
    ("storms", "enabled"): _ENABLED_PERSONALIZABLE,
    ("storms", "expiry_days"): SliderSpec(min=0, max=60, step=1, suffix=" days expiry", personalizable=True),
    ("storms", "popup_fontsize"): _FONTSIZE,
    # Waves is a Weather-tab checkbox, not a Climate radio: it's GFS-sourced (like
    # wind/jetstream, same forecast cycle) rather than RTOFS like currents, and unlike
    # currents/sst/temperature/ozone/stormwatch it doesn't cover landmasses, so it has
    # much less potential to visually clash with another base layer shown alongside it.
    ("waves", "enabled"): _ENABLED_PERSONALIZABLE,
    ("waves", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("waves", "palette"): SelectSpec([
        ("ocean_storm", "Ocean storm"),
        ("neon_surge", "Neon surge"),
        ("solar_flare", "Solar flare"),
    ], personalizable=True),
    ("waves", "opacity"): _OPACITY,
    ("waves", "min_wave_height"): SliderSpec(
        min=0, max=5, step=0.25, suffix=" m", zero_label="off", personalizable=True
    ),
    ("waves", "particle_speed"): _PARTICLE_SPEED_LIKE,
    ("waves", "particle_size"): _PARTICLE_SIZE,
    ("waves", "bar_length"): SliderSpec(min=1, max=8, step=1, personalizable=True),
    ("waves", "particle_opacity"): _PARTICLE_OPACITY,
    ("waves", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    # --- Climate (sst, currents, temperature, ozone, stormwatch) ---
    # Since #312, sst's palette/min_c/max_c/opacity all apply entirely client-side (see
    # tasks/sst.py's _mode_settings_signature docstring) -- personalizable across the
    # board. mode is safe too: both modes render every cycle regardless of which is
    # configured (SSTUpdater.run()), so switching is instant, same as species/mode below.
    ("sst", "enabled"): _ENABLED_PERSONALIZABLE,
    ("sst", "mode"): _MODE_OPTIONS,
    ("sst", "opacity"): _OPACITY,
    ("sst", "palette"): SelectSpec([
        ("thermal", "Thermal"),
        ("vivid", "Vivid"),
        ("deep", "Deep"),
        ("ocean", "Ocean"),
    ], personalizable=True),
    ("sst", "min_c"): _MIN_MAX_C,
    ("sst", "max_c"): _MIN_MAX_C,
    ("sst", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("currents", "enabled"): _ENABLED_PERSONALIZABLE,
    ("currents", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("currents", "palette"): SelectSpec([
        ("thermal_red", "Thermal red"),
        ("electric_blue", "Electric blue"),
        ("toxic_neon", "Toxic neon"),
        ("cyberpunk", "Cyberpunk"),
    ], personalizable=True),
    ("currents", "opacity"): _OPACITY,
    ("currents", "particle_speed"): _PARTICLE_SPEED_LIKE,
    ("currents", "particle_opacity"): _PARTICLE_OPACITY,
    # NOT personalizable, unlike this section's other settings: CurrentsUpdater.plot()
    # reads this server-side to mask out slow-water cells (NaN -> alpha 0) BEFORE
    # encoding the shared velocity texture -- personalizing it would need a per-user
    # server re-render, which this architecture doesn't do (unlike palette/opacity/
    # fill_floor/fill_knee, which only remap the SAME shared texture client-side).
    ("currents", "current_speed_minimum"): SliderSpec(
        min=0.0, max=5.0, step=0.1, decimals=2, suffix=" m/s"
    ),
    ("currents", "trail_length"): _TRAIL_LENGTH,
    ("currents", "trail_thickness"): _TRAIL_THICKNESS,
    ("currents", "fill_floor"): SliderSpec(min=0.0, max=1.0, step=0.05, decimals=2, suffix=" m/s", personalizable=True),
    ("currents", "fill_knee"): SliderSpec(min=0.0, max=2.5, step=0.05, decimals=2, suffix=" m/s", personalizable=True),
    ("currents", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("temperature", "enabled"): _ENABLED_PERSONALIZABLE,
    ("temperature", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("temperature", "opacity"): _OPACITY,
    ("temperature", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("ozone", "enabled"): _ENABLED_PERSONALIZABLE,
    ("ozone", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("ozone", "palette"): SelectSpec([
        ("alert", "Alert (magenta -> yellow)"),
        ("high_contrast", "High contrast (red -> pale yellow)"),
    ], personalizable=True),
    ("ozone", "critical_du"): SliderSpec(min=150.0, max=500.0, step=10.0, decimals=1, suffix="du", personalizable=True),
    ("ozone", "opacity"): _OPACITY,
    ("ozone", "stormwatch"): ToggleSpec(personalizable=True),
    ("ozone", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    ("stormwatch", "enabled"): _ENABLED_PERSONALIZABLE,
    ("stormwatch", "level_of_detail"): _LEVEL_OF_DETAIL,
    ("stormwatch", "min_cape"): SliderSpec(min=0, max=5000, step=100, suffix="J/Kg", personalizable=True),
    ("stormwatch", "opacity"): _OPACITY,
    ("stormwatch", "cache_expiry_days"): _CACHE_EXPIRY_DAYS,
    # --- Greenhouse gases (CO2/CH4 -- Absolute from GEOS-CF, Anomaly computed against
    # a CAMS EGG4 baseline year). Per-species scale/palette settings are flat,
    # species-prefixed keys (co2_min_ppm, ch4_palette, ...) rather than a nested dict --
    # see tasks/greenhouse_gases.py's _SCALE_SETTING_KEYS for why. Since #312, all of
    # species/mode/opacity/min/max/palette apply entirely client-side (see
    # tasks/greenhouse_gases.py's plot() docstring) -- personalizable across the board,
    # EXCEPT baseline_year, which selects which EGG4 baseline file gets fetched/diffed
    # server-side (a genuine backend behaviour change, not just display). ---
    ("greenhouse_gases", "enabled"): _ENABLED_PERSONALIZABLE,
    ("greenhouse_gases", "species"): _GHG_SPECIES,
    ("greenhouse_gases", "mode"): _MODE_OPTIONS,
    ("greenhouse_gases", "baseline_year"): _GHG_BASELINE_YEAR,
    ("greenhouse_gases", "opacity"): _OPACITY,
    ("greenhouse_gases", "co2_min_ppm"): SliderSpec(min=380, max=450, step=1, suffix=" ppm", personalizable=True),
    ("greenhouse_gases", "co2_max_ppm"): SliderSpec(min=380, max=450, step=1, suffix=" ppm", personalizable=True),
    ("greenhouse_gases", "co2_palette"): _GHG_PALETTE,
    ("greenhouse_gases", "ch4_min_ppb"): SliderSpec(min=1600, max=2100, step=10, suffix=" ppb", personalizable=True),
    ("greenhouse_gases", "ch4_max_ppb"): SliderSpec(min=1600, max=2100, step=10, suffix=" ppb", personalizable=True),
    ("greenhouse_gases", "ch4_palette"): _GHG_PALETTE,
    # Only `enabled`/`variable` are personalizable for air_quality: every variable
    # always renders every cycle (AirQualityUpdater.run()), so switching which one is
    # DISPLAYED is instant -- but opacity and every *_min threshold ARE baked
    # server-side into the shared render's alpha feathering (AirQualityUpdater.plot()),
    # so personalizing them would need a per-user re-render, unlike SST/GHG post-#312.
    ("air_quality", "enabled"): _ENABLED_PERSONALIZABLE,
    ("air_quality", "variable"): _AQ_VARIABLE,
    ("air_quality", "opacity"): SliderSpec(min=0, max=100, step=1),
    # Only a MINIMUM ("highlight above this") is user-configurable, not a max --
    # tasks/air_quality.py's _FIXED_CEILING holds each variable's non-configurable
    # gradient top. An independent min+max pair used to invite a scale (e.g. min=1,
    # max=5) that clipped nearly the entire globe to the gradient's bottom colour,
    # indistinguishable from "the layer is broken". Each slider's max here matches
    # its variable's _FIXED_CEILING exactly -- a min above that would hide everything.
    ("air_quality", "pm2_5_min"): SliderSpec(min=0, max=250, step=5, suffix=" µg/m³"),
    ("air_quality", "pm10_min"): SliderSpec(min=0, max=400, step=5, suffix=" µg/m³"),
    ("air_quality", "aod_min"): SliderSpec(min=0, max=3, step=0.1),
    # General (all-sources) SO2 -- its own threshold, independent of
    # ("volcanoes", "so2_min") above, which now belongs to the separate
    # volcanic-specific SO2 variable Smoke Plume renders instead.
    ("air_quality", "so2_min"): SliderSpec(min=0, max=20, step=0.5, decimals=1, suffix=" DU"),
    # --- Flood Risk (issue #371) -- both mode's variants render every cycle
    # regardless of the configured mode (FloodRiskUpdater.run(), same "render
    # everything, publish only what's selected" shape as GHG's species/mode), and
    # rendering is entirely client-side (raw data texture + client LUT, issue #312's
    # convention) -- so mode/opacity are personalizable exactly like GHG's above. ---
    ("flood_risk", "enabled"): _ENABLED_PERSONALIZABLE,
    ("flood_risk", "mode"): _FLOOD_RISK_MODE,
    ("flood_risk", "opacity"): _OPACITY,
    # --- Background (shipping_collector, lightning_collector, satellites_collector,
    # data_collector, housekeeper) ---
    ("shipping_collector", "enabled"): _ENABLED,
    ("shipping_collector", "listen_duration"): _LISTEN_DURATION_MINUTES,
    ("shipping_collector", "sleep_interval"): _SLEEP_INTERVAL_MINUTES,
    ("shipping_collector", "vessel_track_expiry_days"): _VESSEL_TRACK_EXPIRY_DAYS,
    ("shipping_collector", "log_level"): _LOG_LEVEL,
    ("lightning_collector", "enabled"): _ENABLED,
    ("lightning_collector", "sleep_interval"): _SLEEP_INTERVAL_MINUTES,
    ("lightning_collector", "expiry_hours"): _HOURS,
    ("lightning_collector", "log_level"): _LOG_LEVEL,
    # AircraftCollector's cache-warming sweep (issue #215) -- requests_per_minute is
    # the whole collector's adsb.lol budget, shared across hotspot and background
    # sampling alike (see GlobalSampleScheduler). aircraft_track_expiry_hours is
    # hours-scale (not shipping_collector's days-scale vessel_track_expiry_days),
    # since flights last hours, not days.
    ("flightradar_collector", "requests_per_minute"): SliderSpec(min=1, max=60, step=1, suffix="/min"),
    ("flightradar_collector", "starvation_floor_minutes"): SliderSpec(min=1, max=120, step=1, suffix=" min"),
    ("flightradar_collector", "coarse_grid_deg"): SliderSpec(min=5, max=90, step=5, suffix=" deg"),
    ("flightradar_collector", "interest_max_age_seconds"): SliderSpec(min=5, max=120, step=5, suffix="s"),
    ("flightradar_collector", "aircraft_track_expiry_hours"): SliderSpec(
        min=0, max=72, step=1, suffix=" hour", zero_label="Never", pluralize=True
    ),
    ("flightradar_collector", "log_level"): _LOG_LEVEL,
    ("satellites_collector", "enabled"): _ENABLED,
    ("satellites_collector", "groups"): _CELESTRAK_GROUPS,
    ("satellites_collector", "log_level"): _LOG_LEVEL,
    # data_collector.datasources is deliberately NOT here -- see
    # render_datasources_accordion in _field_macros.html.
    # data_collector.channel_enabled is deliberately NOT here either -- it's a per-source
    # data-acquisition opt-out (independent of any layer's frontend `enabled`), rendered
    # on the Data Status page rather than as a generic config-tab field.
    ("data_collector", "enabled"): _ENABLED,
    ("data_collector", "backfill_poll_seconds"): SliderSpec(min=10, max=600, step=10, suffix="s"),
    ("data_collector", "cache_hours"): _HOURS,
    ("data_collector", "log_level"): _LOG_LEVEL,
    ("housekeeper", "enabled"): ToggleSpec(),
    ("housekeeper", "days_between_runs"): SliderSpec(
        min=1, max=14, step=1, prefix="every ", suffix=" day", pluralize=True
    ),
    ("housekeeper", "field_expiry_hours"): _HOURS,
    ("housekeeper", "dry_run"): ToggleSpec(),
    ("housekeeper", "log_level"): _LOG_LEVEL,
}

# Per-(section, option) label overrides, ported from the legacy JS's customLabelText
# special cases -- only the ones relevant to fields with a FIELD_SPECS entry so far.
_LABEL_OVERRIDES = {
    ("animation", "forecast_stepping"): "Forecast stepping (hourly playback)",
    ("animation", "stepping_rate"): "Forecast stepping rate",
    ("quakes", "min_mag"): "Minimum magnitude",
    ("stormwatch", "min_cape"): "Minimum CAPE Threshold",
    ("currents", "fill_floor"): "Fill Floor (min speed shown)",
    ("currents", "fill_knee"): "Fill Knee (full-opacity speed)",
    ("wind", "vector_color"): "Particle color",
    # particle_opacity needs no override -- field_label()'s generic "spaced +
    # capitalised" fallback already produces "Particle opacity" from the option name.
    ("wind", "opacity"): "Heatmap opacity",
    ("currents", "opacity"): "Heatmap opacity",
    ("ozone", "critical_du"): "Critical Ozone Threshold (Dobson Units)",
    ("pwat", "critical_pwat"): "Critical Moisture Threshold (mm)",
    ("satellites_collector", "groups"): "Satellite groups (CelesTrak)",
    ("fires", "opacity"): "Heatmap opacity",
    ("fires", "min_risk_display"): "Heatmap minimum fire risk",
    ("fires", "min_risk_filter"): "Fire risk display threshold",
    ("volcanoes", "smoke_opacity"): "Smoke Plume Opacity",
    ("volcanoes", "so2_min"): "Smoke Plume Threshold",
    ("world_events", "min_mentions"): "Minimum corroborating sources",
    ("world_events", "backfill_days"): "Initial backfill window",
    ("world_events", "show_warfare"): "Show conflict",
    ("world_events", "show_targeted_violence"): "Show targeted / mass violence",
    ("troublespots", "cell_size_deg"): "Cell size (degrees)",
    ("troublespots", "window_hours"): "Convergence window",
}


def field_label(section: str, option: str) -> str:
    override = _LABEL_OVERRIDES.get((section, option))
    if override is not None:
        return override
    spaced = option.replace("_", " ")
    return spaced[:1].upper() + spaced[1:]


# Friendly section headings for the "X Properties" title above each settings block --
# the same name used for that layer's toggle/radio in the Show tab, so both parts of
# the UI call a layer by the same name instead of the raw config key in brackets
# (e.g. "Precipitable Water Properties", not "[pwat] Properties").
SECTION_LABELS = {
    "clouds": "Clouds",
    "isobars": "Isobars",
    "wind": "Wind",
    "precipitation": "Precipitation",
    "pwat": "Precipitable Water",
    "lightning": "Lightning",
    "storms": "Storm Track",
    "sst": "Sea Surface Temp",
    "greenhouse_gases": "Greenhouse Gases",
    "air_quality": "Air Quality",
    "currents": "Ocean Currents",
    "jetstream": "Jet Stream",
    "waves": "Waves",
    "temperature": "Air Temperature",
    "ozone": "Ozone",
    "stormwatch": "Storm Watch",
    "quakes": "Earthquakes",
    "volcanoes": "Volcanoes",
    "fires": "Wildfires",
    "world_events": "World Events",
    "troublespots": "Troublespots",
    "satellites": "Satellites",
    "terminator": "Terminator Night/day Shade",
    "markers": "Place Markers",
    "flightradar": "Flight Radar",
    "landmass": "Landmass Outlines",
    "shipping": "Shipping",
    "shipping_collector": "Shipping Collector (AIS Loop)",
    "lightning_collector": "Lightning Collector Daemon",
    "satellites_collector": "Satellites Collector",
    "data_collector": "Data Collector",
}


def section_label(section: str) -> str:
    """Friendly "X Properties" heading for a settings section -- matches the Show
    tab's label for that layer's toggle/radio exactly. Sections with no Show-tab entry
    (map_builder, animation, housekeeper) fall back to a title-cased, space-split
    version of the section key."""
    return SECTION_LABELS.get(section, section.replace("_", " ").title())


def to_display_value(spec: SliderSpec, raw_value):
    """Converts a stored value into the space the HTML slider actually operates in.
    Only clouds.threshold uses this today (raw 0-255 byte, displayed/edited as a
    0-100 percentage) -- everything else is a no-op."""
    if not spec.byte_to_percent:
        return raw_value
    try:
        raw = float(raw_value)
    except (TypeError, ValueError):
        raw = 0
    return round((raw / spec.raw_max) * spec.max)


def clamp_slider_value(spec: SliderSpec, value) -> float:
    """A stored value outside [min, max] (e.g. left over from a range that has since
    been corrected, or from an unvalidated write predating validate_against_specs)
    would otherwise make the rendered badge and the range input's clamped position
    disagree. Clamping once, before either is rendered, keeps them consistent.

    Whole-step sliders (min_cape, runs_per_day, fontsize, ...) match legacy JS
    branches that used parseInt for the badge -- always a clean int ("45"), never
    "45.0". Returning an int here (rather than float(value)'s float) means
    format_slider_badge doesn't need to re-derive that on every call."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = spec.min
    else:
        v = max(spec.min, min(spec.max, v))
    if float(spec.step).is_integer():
        return int(round(v))
    return v


def format_slider_badge(spec: SliderSpec, value) -> str:
    if spec.zero_label is not None:
        try:
            if float(value) == 0:
                return spec.zero_label
        except (TypeError, ValueError):
            pass

    base = str(value) if spec.decimals is None else f"{float(value):.{spec.decimals}f}"
    suffix = spec.suffix
    if spec.pluralize:
        try:
            count = float(value)
        except (TypeError, ValueError):
            count = None
        if count is not None and count != 1:
            suffix = f"{suffix}s"
    return f"{spec.prefix}{base}{suffix}"


def is_long_or_url_field(option: str, value) -> bool:
    """Ported from the legacy JS's fallback branch: a long value or a *url*-named
    option renders full-width instead of the default half-width column."""
    return len(str(value)) > 35 or "url" in option


def is_api_key_field(option: str) -> bool:
    """Secrets injected by AtmosGLConfig._inject_secrets (e.g. common.api_key,
    shipping_collector.api_key) render read-only, matching the legacy JS."""
    return "api_key" in option


def validate_against_specs(payload: dict) -> list[str]:
    """Check payload values with a FIELD_SPECS entry against that spec. Fields
    without an entry are left untouched -- same permissive behaviour as today."""
    errors = []
    for (section, option), spec in FIELD_SPECS.items():
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict) or option not in section_payload:
            continue
        value = section_payload[option]

        if spec.kind == "slider":
            try:
                v = float(value)
            except (TypeError, ValueError):
                errors.append(f"{section}.{option}: expected a number, got {value!r}")
                continue
            # byte_to_percent fields are posted in raw/stored space (0-255), not the
            # displayed slider's 0-100 percent space.
            hi = spec.raw_max if spec.raw_max is not None else spec.max
            if not (spec.min <= v <= hi):
                errors.append(f"{section}.{option}: {v} outside [{spec.min}, {hi}]")
        elif spec.kind == "select":
            # Some stored values are ints even though option values are declared as
            # strings -- compare as strings, like the rendering
            # macro's "selected" check, so a legitimate value isn't rejected.
            valid = {str(opt_value) for opt_value, _ in spec.options}
            if str(value) not in valid:
                errors.append(
                    f"{section}.{option}: {value!r} not one of {sorted(valid)}"
                )
        elif spec.kind == "multiselect":
            valid = {str(opt_value) for opt_value, _ in spec.options}
            if not isinstance(value, list) or not all(str(v) in valid for v in value):
                errors.append(
                    f"{section}.{option}: {value!r} not a subset of {sorted(valid)}"
                )
        elif spec.kind == "grouped_transfer":
            valid = {str(opt_value) for _, opts in spec.groups for opt_value, _ in opts}
            if not isinstance(value, list) or not all(str(v) in valid for v in value):
                errors.append(
                    f"{section}.{option}: {value!r} not a subset of {sorted(valid)}"
                )
        elif spec.kind == "toggle":
            if not isinstance(value, bool):
                errors.append(f"{section}.{option}: expected true/false, got {value!r}")
        # ColorSpec: matches the legacy JS's permissiveness -- any string is accepted
        # (colors are freeform hex/name text, not a closed option set).

    return errors
