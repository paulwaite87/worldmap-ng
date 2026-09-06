# Domain context — atmos-gl

Domain language for the map's data layers. Extend as terms are sharpened; keep
definitions to one sentence.

## Layers

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Scalar field** | A layer rendered as a single-scalar `contourf` heatmap over a value range — temperature, ozone, and stormwatch (CAPE) — sharing one renderer (`ScalarFieldUpdater`) and differing only by a `ScalarFieldSpec` (colormap, range, `extend`, key ticks, title). | scalar layer, heatmap layer |

A **Scalar field** is distinct from the vector layers (wind, currents), the
boundary/level layers (isobars, precipitation's `BoundaryNorm`), and SST's
runtime-computed range — those do not share the scalar-field renderer or its spec
shape.

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Single-hour scalar updater** | `SingleHourScalarUpdater` (`tasks/single_hour_scalar.py`), the shared base behind `ScalarFieldUpdater` and `PrecipitationUpdater` — owns the constructor fields and `run()` wiring (`get_gfs_state()` warm-up → key-refresh hook → `render_all_hours()`) genuinely identical between the two. `plot()` stays a full per-subclass override, not hook-split: the two classes' static-render color models and texture pipelines diverge too much (precipitation's discrete banded palette + smoothing/floor/sqrt-transform pipeline vs. the scalar-field trio's plain `Normalize`/threshold `cmap`) for splitting `plot()` into sub-hooks to pay off — see the module's own docstring. | scalar renderer base |
| **Troublespot** | A live-computed, no-table-of-its-own layer (`db/troublespot_adapter.py`, `lib/troublespot_contours.py`) flagging areas where 2+ of a fixed 4-type roster — World Events (one type regardless of its own category), Earthquakes, Fires, Volcanic Activity — converge within a configurable geographic cell (`cell_size_deg`) and time window (`window_hours`). Not a collector: it has no periodic task, no `channel_key`, no Data Status row of its own — `lib/layer_availability.py`'s Show-toggle gate instead requires `MIN_CONVERGENCE_TYPES` (2) of its 4 source collectors' channels to be enabled, since fewer can never satisfy the convergence rule. Bands (Elevated/High/Severe, by distinct-type count 2/3/4) are nested contour polygons: each band's binary inclusion mask is Gaussian-smoothed *before* contouring (never the raw integer counts — smoothing those would dilute a single high-count cell below its own band), then contoured at the 0.5 level via `contourpy`, giving a smooth boundary without ambiguity in band membership itself. | convergence layer, hotspot (avoid — collides with Flight Radar's unrelated viewport-priority-sampling "hotspot") |
| **Flood Risk** | A land-only layer (`tasks/flood_risk.py`, issue #371 and its follow-up grilling) with two independently-sourced modes sharing one `flood_risk` section/output path, not two views of the same data: **Live** (`FloodRiskLiveCollector`, `collectors/flood_risk.py`) is NASA LANCE MODIS observed flood detection ("Observed Current Inundation") — a binary flood band (0/1) rebuilt from up to 287 10x10deg GeoTIFF tiles every cycle a tile changes or expires, needing `EARTHDATA_TOKEN`; **Historical** (`FloodRiskHistoricalCollector`) is Copernicus/JRC's static Global River Flood Hazard map (100-year return period) mosaicked once from per-tile downloads and cached forever, classified into a fixed 0-4 depth-hazard category, no credential needed. Live originally sourced a Copernicus GloFAS ensemble river-discharge FORECAST via EWDS (2yr/5yr/20yr severity bands) but was abandoned entirely — not just re-patched — after a real OOM crash, a second distinct OOM discovered even after fixing the first, and unfixable network flakiness against ECMWF/Copernicus's shared object-store backend (see `collectors/flood_risk.py`'s module docstring for the full history). The two bands remain deliberately NOT comparable (different scales, different physical meaning) — each keeps its own encode domain/palette/legend, both server-side (`_LIVE_ENCODE_DOMAIN`/`_HISTORICAL_ENCODE_DOMAIN`) and client-side (`ui/modules/flood_risk.js`'s `ENCODE_DOMAIN`/`LIVE_COLORS`/`HISTORICAL_COLORS`). `FloodRiskUpdater` renders both variants every cycle regardless of the configured mode (both are now single-shot cached-mosaic renders, not per-forecast-hour series) and publishes only the selected one to the canonical output path (the same "render everything, publish only what's selected" shape `GhgUpdater` established for species/mode) — so a mode switch applies on the frontend's next poll tick, no render wait, and the `EARTHDATA_TOKEN` gate (Live-mode-only; Historical needs no credential) never disables the whole section, only Live. | flood layer, MODIS flood layer |

## Backend collectors

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Single-file field collector** | A `FieldCollectorBase` subclass (`SingleFileFieldCollector`, `collectors/field_base.py`) that fetches one whole file per forecast hour for a single product — `GfsWavesCollector`, `RtofsCurrentsCollector` — sharing one `collect()`/`backfill_hour()` implementation behind `_resolve_download_url()`/`_guard_cycle()` hooks. Distinct from `GfsAtmosCollector`'s multi-product byte-range fetch, which stays its own implementation, subclassing `FieldCollectorBase` directly. | multi-file collector |
| **Collector driver** | `CollectorDriver` (`collectors/driving.py`), the shared gate/try/record envelope behind `EventFeedDriver` (event feeds + file caches — `is_stale`/`has_new_data`/`collect`, own `last_runs` timestamp bookkeeping) and `FieldCollectorDriver` (the three `FieldCollectorBase` subclasses — unconditional `collect(ctx)`, freshness handled internally per forecast hour, a shared `CycleContext`). `_drive_one()` stays a full per-subclass override, not hook-split further: construction arity, whether there's an external freshness pre-check, and what per-cycle state gets threaded through are real domain differences between the two families, not incidental duplication — see the module's own docstring and `docs/adr/0001-dont-unify-gfs-rtofs-baseline-probing.md` for the same reasoning applied to a different pair in this package. | driver base |
| **World Events** | `WorldEventsCollector` (`collectors/world_events.py`), a point-marker event feed sourced from the GDELT Event Database 2.0, filtered at ingest time to a curated CAMEO event-code allowlist across four categories — Explosion, Conflict, Targeted/mass violence, and Diplomacy (root-04 events matched against a curated organization-name list, not any diplomatic contact) — rather than GDELT's full 300+-code taxonomy. Coverage of the last `backfill_days` is maintained every cycle by comparing the oldest stored event against that window and walking GDELT's `masterfilelist.txt` for whatever gap remains, not a one-shot empty-table seed. | GDELT events, conflict layer |

## Data conventions

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Direction convention (FROM)** | Wave/wind direction fields (GRIB `dirpw`/`mwd`) are WMO convention: the angle a flow arrives FROM, not the heading it travels TOWARD. Deriving a travel vector requires negating: `u = -mag*sin(dir)`, `v = -mag*cos(dir)` (see `lib/unpack.py`'s `_swell_uv`). | heading, bearing |

Getting this backwards silently points every particle/vector layer 180° from its true
direction — a real bug (`waves_data_unpack`) lived exactly here before being fixed and
pinned by `tests/test_lib_unpack.py`.

## Backend render tasks

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **ForecastState** | The (run_date, run_id, forecast_hour) triple a render call operates on (`tasks/common.py`), passed explicitly everywhere — never cached as mutable instance state. Built via `Updater.get_gfs_state()`/`get_rtofs_state()` (the shared per-cycle baseline) or `ForecastState.at_hour(run_date, run_id, fhour)` (a specific catalog hour). | run state, forecast context |

Every `Updater`/`MultiHourRenderMixin` method that needs to know "which run, which
hour" takes a `ForecastState` parameter rather than reading `self`. Before this, the
four raw attributes it replaced (`run_date_str`/`run_id`/`forecast_hour_str`) were
mutated directly on `self`, forcing `hasattr` guards and two separate save/restore
`try`/`finally` dances to avoid callers clobbering each other's state.

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Land mask cache** | `LandMaskCache` (`lib/coastline.py`), a per-run, per-grid-shape cache around `coastline_land_mask()` — shared by `CurrentsUpdater` and `WavesUpdater`, whose own caching wrappers used to be byte-identical. Paired with `nearest_fill_and_regrid_uv()` (nearest-fill native NaN, then regrid u/v) — also byte-identical between the two before extraction, apart from the regrid-step constant. Each caller applies its own steps (e.g. currents' speed-minimum threshold) and the land-mask cut itself afterward, since ordering differs between the two. Distinct from `coastline_land_mask()` itself, which `sst.py`/`greenhouse_gases.py` call directly with no caching wrapper. | coastline cache |

## Frontend legend wiring

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Standard legend** | `standardLegend(slotId, outfileFor, opacityFallback)` (`ui/modules/_legend.js`), the `showLegend`+`opacityUniform` wiring every layer module rebuilt independently (12 near-identical copies, each also computing `keyFilename` a second time for its own separate `keyUrl` chase property). `outfileFor(cfg)` lets a caller insert its own variant suffix first (`sst.js`'s mode, `air_quality.js`'s variable, `greenhouse_gases.js`'s species+mode) via `insertBeforeExtension` — the same "insert before extension" split `keyFilename` itself now delegates to, previously re-derived 3× by those same three modules. Omitting `opacityFallback` (not passing a fallback number at all) preserves `currents.js`/`jetstream.js`'s documented exception — their legend key stays visible independent of the fill's own opacity — rather than defaulting to always-computed opacity gating. | legend helper |

## Frontend popups

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Popup content block** | A typed entry in `buildPopupHtml`'s (`ui/modules/_feedhelpers.js`) `blocks` array — `divider`, `rows`, `line`, `emphasis`, `notice`, or `fallback` — each a named, reusable shape rather than a raw-HTML escape hatch, so every current popup layout (a fused title line, a `<br>`-separated field list, a conditional route callout, a live-computed row colour, a no-data fallback) stays expressible without a caller reaching around the shared model. `line` and `rows` are both block-level (wrapped in a `<div>`), so a block always starts on its own line regardless of what precedes it — no reliance on a preceding divider or trailing `<br>`. | popup section |
| **Title variant** | A named entry in `buildPopupHtml`'s `TITLE_VARIANTS` (`default`/`callsign`/`alert`/`plain`/`fire`) fixing a title's color+size as one unit, rather than raw `titleColor`/`titleSize` params a caller could set to anything. Add a new variant when a genuinely distinct, deliberately-chosen style shows up (e.g. `fire`'s `#ff5a1f`, found only when migrating fires.js — a 9th popup consumer missed in the original cataloguing pass); don't fold a real distinct color into an existing variant just to avoid naming one. | title style |

`buildPopupHtml` (content) and the widened `hoverPopup` (`_hoverpopup.js` — show/hide/
positioning, now accepting `layerId` as a string-or-array, a configurable `event`
("enter"/"move"), and a live `enabled` predicate) together are the "one-stop-shop"
every popup-bearing layer goes through, including `markers.js` — architecture review
candidate #6, which superseded `docs/adr/0002-dont-extend-hoverpopup-for-markers.md`
(that ADR's four axes — multi-layer, mousemove, live-enable, a caller-specific
`maxWidth` — are exactly what `hoverPopup` widened to cover). `popupCard`, the prior
content model, is deleted; every caller migrated onto `buildPopupHtml` instead.
