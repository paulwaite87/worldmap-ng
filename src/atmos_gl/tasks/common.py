#!/usr/bin/env python3
import os
import json
import logging
import shutil
import numpy as np
from cartopy.util import add_cyclic_point
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Internal library import
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.db.region_adapter import RegionAdapter
from atmos_gl.lib import fieldstore
from atmos_gl.db.process_status_adapter import ProcessStatusAdapter
from atmos_gl.lib.output_files import OUTFILES
from atmos_gl.lib.data_status import (
    freshness_percent,
    estimate_next_update,
    period_s_from_runs_per_day,
    read_process_status,
    resolve_run_epoch_utc,
    build_status,
)

logger = logging.getLogger(__name__)

# Seconds between layer_builder's fan-out cycles (every cycle dispatches every updater).
# Canonical home is here, not layer_builder.py, so Updater.layer_status() can use it for
# next_update without layer_builder importing tasks.common creating a cycle the other way.
# layer_builder.py imports this rather than defining its own copy.
LAYER_CYCLE_SECONDS = 15

# Upper bound on points in a regrid_for_lod() output grid (~32MB per float64 array at
# the cap). regrid_for_lod's LOD step sizes are tuned to stay comfortably under this at
# world-view scale (the dominant case: the frontend always projects onto a globe), but
# an earlier tuning (0.05/0.125/0.25 degrees, sized for a regional view) applied
# unscaled to a world-view bbox (360x180 degrees) ballooned "high" to ~26M points per
# array and reliably OOM-killed the render worker under concurrent load. This budget is
# a backstop for that failure mode, not the primary mechanism — regrid_for_lod scales
# its step up (coarser) only if the clipped region is large enough to exceed it anyway.
_MAX_LOD_GRID_POINTS = 4_000_000

# Pixel dimensions of the static-fallback PNG Plot.get_figure() renders for every layer
# (used only when forecast_stepping is off or the browser's WebGL path fails -- the
# primary animated rendering decodes raw field data on the GPU at its own resolution,
# independent of this). Always 2:1 since MapData always resolves the global bbox. Used
# to be a user-facing "Target Geometry" setting; dropped since it only ever affected
# this rarely-seen fallback.
STATIC_FALLBACK_GEOMETRY = "4096x2048"


def stringify_bbox(bbox):
    """
    Converts a bbox list into a filename-safe string.
    Example: [-180.0, -90.0, 180.0, 90.0] -> "180.0W_90.0S_180.0E_90.0N"
    Or simpler: "lon-180.0_lat-90.0_lon180.0_lat90.0"
    """
    if not bbox or len(bbox) != 4:
        return "global"

    labels = ["w", "s", "e", "n"]
    return "_".join(f"{labels[i]}{abs(bbox[i]):.1f}" for i in range(4))


def get_bbox_center(bbox):
    """
    Returns the center (longitude, latitude) for a given bbox.
    bbox: [lon_min, lat_min, lon_max, lat_max]
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # Center Latitude is a straight average
    center_lat = (lat_min + lat_max) / 2

    # Center Longitude
    # Handle the Date Line: if the span is negative or crosses 180
    delta_lon = lon_max - lon_min
    center_lon = lon_min + (delta_lon / 2)

    # Normalize longitude to stay within [-180, 180]
    if center_lon > 180:
        center_lon -= 360
    elif center_lon < -180:
        center_lon += 360

    return center_lon, center_lat


class MapRegion:
    def __init__(
        self,
        target_geometry: str | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        region: str | list[float] | None = None,
    ):
        self.region = region
        self.region_identifier = "region"
        self.target_width = target_width
        self.target_height = target_height
        self.region_geometry = target_geometry
        # Solve inter-dependency of these dimensions; explicit dims get
        # priority over the composite geometry string
        if isinstance(target_width, int) and isinstance(target_height, int):
            self.target_geometry = f"{self.target_width}x{self.target_height}"
        elif target_geometry and "x" in target_geometry:
            self.target_width = int(target_geometry.split("x")[0])
            self.target_height = int(target_geometry.split("x")[1])
        self.bbox = None
        self.world_view = False
        self.centre_latitude = 0.0
        self.centre_longitude = 0.0
        self.set_map_region_data(region)

    def set_map_region_data(self, region: str | list[float] | None):
        bbox = None
        bbox_prefix = "region_"
        self.world_view = False

        # Handle explicit 'falsy' regions (None, empty string)
        if not region:
            bbox = [-180.0, -90.0, 180.0, 90.0]
            self.world_view = True
            bbox_prefix = "bbox_"

        elif str(region).startswith("["):
            try:
                data = json.loads(str(region))
                if isinstance(data, list) and not data:
                    bbox = [-180.0, -90.0, 180.0, 90.0]
                    self.world_view = True
                    bbox_prefix = "global_"
                else:
                    bbox = [float(x) for x in data]
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Invalid BBox format for '{region}': {e}")

        else:
            # Database lookup
            region_adapter = RegionAdapter()
            bbox_row = region_adapter.get_region_definition(str(region))
            if bbox_row:
                bbox = [val for _, val in bbox_row.items()]
                bbox_prefix = f"{bbox_prefix}_{region}"
            else:
                logger.warning(
                    f"Region label '{region}' not found; defaulting to global"
                )
                bbox = [-180.0, -90.0, 180.0, 90.0]
                self.world_view = True
                bbox_prefix = "global_"

        # Apply aspect ratio adjustment and 180-degree safety shift
        if bbox:
            self.bbox = bbox
            self.region_identifier = f"{bbox_prefix}_{stringify_bbox(bbox)}"
            self.centre_longitude, self.centre_latitude = get_bbox_center(bbox)


class MapData:
    def __init__(self, config: AtmosGLConfig):
        self.config = config
        self.region = None
        self.shared_state = {}
        self.refresh()

    def refresh(self):
        self.region = MapRegion(target_geometry=STATIC_FALLBACK_GEOMETRY)


@dataclass(frozen=True)
class ForecastState:
    """Which forecast run + hour a render call operates on (GFS or RTOFS; the same
    shape either way). Passed explicitly wherever a render needs to know "when" -- see
    CONTEXT.md's "ForecastState" entry. Two ways to build one:
      * Updater.get_gfs_state()/get_rtofs_state() -- the shared per-cycle NOMADS/RTOFS
        baseline plus a computed now-offset.
      * ForecastState.at_hour(run_date, run_id, fhour) -- a specific catalog hour,
        used when iterating every hour a run has data for (render_all_hours,
        layer_status, and the few callers that resolve their own catalog run).
    """

    run_date_str: str
    run_id: str
    forecast_hour_str: str

    @property
    def fhour(self) -> int:
        return int(self.forecast_hour_str)

    @classmethod
    def at_hour(cls, run_date_str, run_id, fhour) -> "ForecastState":
        return cls(run_date_str, run_id, f"{int(fhour):03d}")


class Updater:
    def __init__(self, config: AtmosGLConfig, section: str, map_data: MapData):
        self.config = config
        self.map_data = map_data
        self.section = section.lower()
        self.settings = config.get_section(self.section)
        self.common = config.get_section("common")
        self.animation = config.get_section("animation")
        self.workdir = self.common.get("workdir", ".")
        # Own, independent store+connection per updater (NOT the shared singleton), so the
        # async fan-out in layer_builder can run updaters concurrently without sharing a
        # psycopg2 connection across threads.
        self._store = fieldstore.make_store(self.workdir)
        self.process_status_adapter = ProcessStatusAdapter()
        # Looked up by section, not user-configurable -- see lib/output_files.py's
        # docstring for why (and for the matching value routes/config.py injects into
        # /api/config so the frontend's cfg.outfile still resolves the same way).
        self.outfile = OUTFILES.get(self.section, "")
        self.output_path = None
        self.enabled = self.settings.get("enabled", False)
        # This is the starting hour (offset) for all renders. It used to be a
        # configurable setting, but since we moved to creating all renders for
        # each hour, and allow the user to play through them, this is not
        # useful to them. We hard-code it to zero here for now.
        self.forecast_hour = 0
        # Per-hour output suffixes a COMPLETE render produces for this layer, relative
        # to the per-hour base (e.g. "isobars_f004"). should_plot_for_hour treats an
        # hour as stale if ANY of these is missing, so deleting (say) a _data.png
        # forces a re-render even when the static .png still exists. Subclasses
        # override this to match what their plot() actually writes. Default: a single
        # static PNG (legacy/plain layers).
        self.per_hour_outputs = [".png"]

        # Fieldstore product this task renders from, for layer_status()'s multi-hour %
        # (see render_all_hours). None (default) means single-shot: sst/clouds/markers
        # don't render per-forecast-hour, so layer_status() falls back to a decaying
        # freshness gauge instead. Multi-hour subclasses (isobars, wind, ...) set this.
        self.status_product: str | None = None

        # Copy map data up to this class for convenience
        self.target_width = map_data.region.target_width
        self.target_height = map_data.region.target_height
        self.world_view = map_data.region.world_view
        self.map_region_identifier = map_data.region.region_identifier
        self.centre_longitude = map_data.region.centre_longitude
        self.centre_latitude = map_data.region.centre_latitude

        # Always set these, which can be over-ridden later if required.
        # If the updater doesn't have an outfile defined, this does nothing.
        self.set_output_path()

    def get_output_path(self) -> str | None:
        return (
            str(os.path.join(self.workdir, self.outfile))
            if self.outfile
            else None
        )

    def set_output_path(self):
        self.output_path = self.get_output_path()
        if self.output_path:
            file_path = Path(self.output_path)
            # Safely verify directories exist for non-image files
            if file_path.suffix not in [".png", ".jpg", ".jpeg"]:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                # Use append mode ('a') to touch/create the file if missing
                with open(self.output_path, "a") as _:
                    pass

    def cache_path(self, filename: str) -> str:
        """Path for a downloadable cache file under the data dir.

        Every cache file carries a uniform '<section>_cache_' prefix. This lets the
        housekeeper find and expire caches by a single marker (no per-layer pattern
        lists to maintain), and makes live render outputs - which never carry the
        prefix - safe from deletion by construction rather than by a guard list.
        """
        return str(
            os.path.join(self.workdir, "data", f"{self.section}_cache_{filename}")
        )

    def remove_output_file(self):
        """Clears the output file of this updater if it exists"""
        output_path = self.get_output_path()
        if output_path and os.path.exists(output_path) and os.path.isfile(output_path):
            os.remove(output_path)

    def _settings_signature(self, values: dict) -> str:
        """Stable hash-equivalent of render-relevant settings, for _is_render_fresh.
        Callers pass just the settings that actually affect the rendered pixels
        (palette, min/max, opacity, key_fontsize, ...) -- not the whole config
        section, so an unrelated setting changing elsewhere doesn't force a
        needless re-render."""
        return json.dumps(values, sort_keys=True)

    def _is_render_fresh(self, out: str, sources: list[str], sig: str) -> bool:
        """Whether `out` is up to date against both its data SOURCES (mtime, the
        original check) and the render-relevant SETTINGS captured in `sig` (from
        _settings_signature). A settings-only change (e.g. a palette edit) touches
        neither `out` nor `sources`' mtimes, so relying on mtime alone silently
        never re-renders until the source data itself next changes -- the
        persisted '<out>.sig' sidecar closes that gap. Missing/mismatched sig
        (including outputs rendered before this check existed) counts as stale
        rather than trusted, so the fix takes effect on next run rather than only
        once the source data happens to refresh."""
        if not os.path.exists(out):
            return False
        if not all(os.path.getmtime(out) >= os.path.getmtime(s) for s in sources):
            return False
        try:
            with open(out + ".sig") as f:
                return f.read() == sig
        except FileNotFoundError:
            return False

    def _write_render_signature(self, out: str, sig: str):
        with open(out + ".sig", "w") as f:
            f.write(sig)

    def _publish_variant(self, variant_output_path: str):
        """Copy a per-variant render (e.g. 'data/sst_anomaly.png', 'data/
        greenhouse_gases_co2_absolute.png', 'data/air_quality_pm2_5.png') to the
        stable, run-agnostic base filename (self.output_path) for anything still
        reading that name directly. Always refreshed each cycle (a cheap file copy),
        independent of whether that variant's plot needed re-rendering this cycle.

        Shared by every layer that renders several variants per cycle and publishes
        only the currently-configured one (SSTUpdater's mode, GhgUpdater's
        species+mode, AirQualityUpdater's variable) -- lifted here after the third
        near-identical copy appeared, rather than adding a fourth."""
        if not os.path.exists(variant_output_path):
            return
        tmp = f"{self.output_path}.tmp"
        shutil.copy2(variant_output_path, tmp)
        os.replace(tmp, self.output_path)

    def _write_meta_sidecar(self, filename: str, entry_key: str, value: dict):
        """Merge `{entry_key: value}` into a small JSON sidecar next to this task's
        output (e.g. 'sst_meta.json', 'ghg_meta.json') and write it back -- used by
        layers whose legend needs a data-dependent scale computed server-side (98th
        percentile of live data) with no other way to reach the client-side canvas
        key. Mirrors WindUpdater's original wind_meta.json precedent; shared here
        after SSTUpdater and GhgUpdater's near-identical versions appeared
        independently (see issue #302)."""
        if not self.output_path:
            return
        meta_path = os.path.join(os.path.dirname(self.output_path), filename)
        try:
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                meta = {}
            meta[entry_key] = value
            with open(meta_path, "w") as f:
                json.dump(meta, f)
        except Exception as e:
            logger.warning(f"{self.section}: could not write {filename}: {e}")

    def get_db_field_at_hour(self, state: "ForecastState", product_name: str) -> dict | None:
        """Fetch a pre-processed field from the fieldstore for a specific forecast run
        + hour. Used by animation frame loops and other multi-hour operations.
        Args:
            state: which run + forecast hour to read.
            product_name: The product name (e.g., "precipitation", "wind")
        Returns:
            Field dict {lat, lon, values, values2, u, v, valid_time} or None
        """
        try:
            fs = self._store
            return fs.get_field(
                state.run_date_str, state.run_id, state.fhour, product_name
            )
        except Exception as e:
            logger.debug(
                f"get_db_field_at_hour({product_name}, f{state.fhour:03d}) failed: {e}"
            )
            return None

    def regrid_for_lod(self, field, lats, lons, fill_value=np.nan, step_override=None):
        """Resample `field` (lat x lon 2D array) onto a finer grid via
        RegularGridInterpolator. Step size is driven by self.level_of_detail (3=high/
        0.15°, 2=medium/0.20°, else low/0.25°), or fixed directly via `step_override`
        for a caller that needs a resolution unrelated to the level_of_detail tiers
        (e.g. SST's coastline-crispness regrid, which needs finer than any LOD tier
        gives and isn't user-configurable). `step_override` bypasses the
        _MAX_LOD_GRID_POINTS budget below entirely -- it's a deliberate, caller-chosen
        fixed resolution, not a tier this method should second-guess by coarsening it.
        Also sets self.lod_desc to the matching "high"/"medium"/"low" string as a side
        effect (some layers log it) -- left unset when step_override is given, since
        none of those labels describe a fixed step.

        Renders are always global now (regions are reporting-only -- see docs/adr/
        0004-render-bbox-clipping-is-dead-code.md), so this no longer clips to a bbox
        first; step sizes are tuned assuming a world-scale field -- "high" lands at
        ~73% of _MAX_LOD_GRID_POINTS at world scale, so normal operation has headroom
        and doesn't routinely hit the cap below. The cap itself still scales the step
        up (coarser) as a backstop if the field is large enough to exceed the budget
        regardless — see that constant's docstring.

        Returns (new_lats, new_lons, field_smooth) — the LOD grid axes and the
        resampled field, ready to hand to contourf.
        """
        if step_override is not None:
            step = step_override
        elif self.level_of_detail == 3:
            step = 0.15
            self.lod_desc = "high"
        elif self.level_of_detail == 2:
            step = 0.20
            self.lod_desc = "medium"
        else:
            step = 0.25
            self.lod_desc = "low"

        if step_override is None:
            lat_span = lats.max() - lats.min()
            lon_span = lons.max() - lons.min()
            estimated_points = (lat_span / step + 1) * (lon_span / step + 1)
            if estimated_points > _MAX_LOD_GRID_POINTS:
                scale = (estimated_points / _MAX_LOD_GRID_POINTS) ** 0.5
                logger.debug(
                    f"{self.section}: LOD grid ({int(estimated_points):,} pts) exceeds "
                    f"budget ({_MAX_LOD_GRID_POINTS:,}); scaling step {step:.3f}° -> "
                    f"{step * scale:.3f}°"
                )
                step *= scale

        new_lats = np.arange(lats.min(), lats.max() + step, step)

        if lats[0] > lats[-1]:
            lats_inc, field_inc = lats[::-1], field[::-1, :]
        else:
            lats_inc, field_inc = lats, field

        # Longitude is cyclic once the native data actually spans the whole globe
        # (renders are always global now -- see this method's docstring above). A
        # value-derived arange(lons.min(), lons.max()+step, step) leaves a partial-step
        # gap at the antimeridian seam whenever the native grid's own wrap doesn't
        # sample a full 360 degrees -- GFS-style grids sample lon 0..359.75 at 0.25
        # deg, which _standardize_lon wraps to -180..179.75 (359.75 degrees, not 360).
        # That gap means the output grid's actual column count doesn't equal 360/step,
        # silently misaligning every column near the seam against the GPU particle
        # shader's texel math (VEL_SAMPLE), which assumes exactly 360/width degrees per
        # column -- found live via waves' animated bars reading valid ocean data ~2
        # columns (~12-14km) west of where the shader thought it was sampling,
        # producing particles that appeared to cross onto land on west-facing coasts.
        # Only treated as cyclic for a genuinely global span -- a caller passing a
        # smaller regional field (this method's own tests do, for speed) keeps the
        # plain value-derived axis below, since there's no real wraparound to fix there.
        lon_span = lons.max() - lons.min()
        if lon_span >= 359.0:
            n_lon = int(round(360.0 / step))
            new_lons = lons.min() + np.arange(n_lon) * step
            # RegularGridInterpolator has no periodic-boundary support, so give it a
            # real neighbour at the seam: a duplicate of the first column, appended at
            # lons[0]+360 -- physically the same meridian, wrapped around. Skipped when
            # the native data already closes the loop itself (lons[-1] already at or
            # past the wrap point), to avoid a non-ascending duplicate point.
            if lons[-1] < lons[0] + 360.0 - 1e-9:
                lons_for_fn = np.concatenate([lons, [lons[0] + 360.0]])
                field_for_fn = np.concatenate([field_inc, field_inc[:, :1]], axis=1)
            else:
                lons_for_fn, field_for_fn = lons, field_inc
        else:
            new_lons = np.arange(lons.min(), lons.max() + step, step)
            lons_for_fn, field_for_fn = lons, field_inc

        fn = RegularGridInterpolator(
            (lats_inc, lons_for_fn), field_for_fn, bounds_error=False, fill_value=fill_value
        )
        mesh_lats, mesh_lons = np.meshgrid(new_lats, new_lons, indexing="ij")
        field_smooth = fn((mesh_lats, mesh_lons))
        return new_lats, new_lons, field_smooth

    def close_lon_seam_for_contour(self, lons, field, lon_span_threshold=359.0):
        """Appends a duplicate of the first column at lons[-1]+step (cartopy.util.
        add_cyclic_point) so a contour()/contourf() call spanning the whole globe
        closes the loop at the antimeridian, instead of leaving a visible seam.

        matplotlib's contour machinery has no concept of a periodic domain: a global
        grid's last column (e.g. 179.75°) and first column (-180°) are geographically
        adjacent but numerically the two opposite EDGES of a plain rectangular grid,
        so contour lines simply stop dead at each edge and filled polygons don't
        connect across them -- caught live as a hard vertical break in both the
        isobars and precipitation static renders, running the full height of the
        antimeridian.

        Deliberately applied only at each contourf/contour call site, immediately
        before the call -- NOT folded into regrid_for_lod's own return contract.
        Several consumers of that grid (GPU texture encoding via encode_frames/
        encode_uv, for the wind/currents/waves/precipitation animated layers) rely on
        its exact "new_lons has exactly 360/step columns" width invariant (see
        regrid_for_lod's own comment on the GFS-grid gap it already corrects for) --
        appending a column there would silently break that invariant again the same
        way the wind/currents/wave particle "west of where the shader thought it was
        sampling" bug did before.

        Returns (lons, field) unchanged for a regional (non-global) span -- there's
        no real wraparound to close, and add_cyclic_point would insert a bogus point
        past the field's real edge."""
        lons = np.asarray(lons)
        lon_span = lons.max() - lons.min()
        if lon_span < lon_span_threshold:
            return lons, field
        field, lons = add_cyclic_point(np.asarray(field), coord=lons)
        return lons, field

    def layer_status(self) -> dict:
        """Read-only snapshot for the Config UI's Data Status tab — the layer-task
        counterpart to CollectorBase.data_status(). Never writes; LayerBuilder records
        process_status after each render cycle (see layer_builder.py's _handle_results).

        Two shapes, depending on status_product:
          * set (multi-hour: isobars, wind, ...) — percent is the fraction of the
            forecast hours ALREADY IN THE CATALOG for status_product, from "now" onward,
            that are fully rendered (should_plot_for_hour false, i.e. every
            per_hour_outputs suffix present and fresh). Hours before "now" are excluded
            even though they're still sitting in the catalog (retained for
            data_collector.cache_hours, which is wider than what's currently
            reachable) — the scrubber's minHour ratchets forward and never exposes them,
            so counting them here would report a layer as further along than a user can
            actually see (see _now_fhour). should_plot_for_hour lives on
            MultiHourRenderMixin, not Updater itself — safe to call here because only
            subclasses that set status_product take this branch, and they always mix in
            that class too. Deliberately bounded by what the underlying collector
            has fetched so far, not the theoretical full forecast window — a layer being
            "100%" of what it currently has to work with is correct, not a defect, when
            the collector itself is still catching up (that's the COLLECTOR's data_status
            to report). next_update here means "next time LayerBuilder re-checks this
            task" (LAYER_CYCLE_SECONDS, its fixed fan-out cadence) rather than "next new
            forecast hour" — there's no single well-defined value for the latter since
            rendering is continuous as hours arrive, but the former is still real and
            worth showing rather than leaving blank. Also returns `segments` -- one
            {"hour": fh, "rendered": bool} per now-onward catalog hour -- plus the
            `run_date`/`run_id` they belong to, so the Data Status page can draw a
            segmented per-hour bar instead of a single solid fill.
          * None (single-shot: sst, clouds, markers) — the same decaying-freshness
            formula CollectorBase.data_status() uses, keyed by this task's own section
            and runs_per_day cadence. next_update falls back to an estimate (now +
            period_s) when this task hasn't completed a cycle yet, same as
            CollectorBase.data_status() — see lib/data_status.py's estimate_next_update.

        self.enabled here is the layer's frontend-visibility flag, not a render
        kill-switch — LayerBuilder.start_scheduler() dispatches every TASK_CLASSES entry
        every cycle regardless of it (gated only by the separate layer_builder.enabled
        master switch, which isn't a per-layer concept). next_update must reflect that
        real, unconditional schedule rather than reporting "disabled" for a layer that is
        in fact still being rendered in the background.
        """
        last_updated, last_error, status = read_process_status(
            self.process_status_adapter, self.section
        )
        detail = last_error
        next_update = None
        # Per-hour rendered/pending breakdown for the Data Status page's segmented
        # progress bar -- None for the single-shot branch (no per-hour concept there),
        # so the frontend falls back to its plain solid-fill bar. run_date/run_id ride
        # alongside so the frontend can label each segment ("Run 18Z f003") without
        # having to reverse-engineer it from the hour alone.
        segments = None
        run_date = None
        run_id = None

        if self.status_product:
            percent = 0.0
            resolved = self.latest_store_run([self.status_product])
            if resolved:
                run_date, run_id, hours = resolved
                now_fhour = self._now_fhour(run_date, run_id)
                hours = [h for h in hours if h >= now_fhour]
                total = len(hours)
                rendered = 0
                segments = []
                for fh in hours:
                    state = ForecastState.at_hour(run_date, run_id, fh)
                    is_rendered = not self.should_plot_for_hour(state, self.status_product)
                    if is_rendered:
                        rendered += 1
                    segments.append({"hour": fh, "rendered": is_rendered})
                percent = 100.0 * rendered / total if total else 0.0
                if not detail:
                    detail = f"{run_date} {run_id}Z: {rendered}/{total} hour(s) rendered"
            next_update = estimate_next_update(last_updated, LAYER_CYCLE_SECONDS, True)
        else:
            period_s = period_s_from_runs_per_day(self.settings.get("runs_per_day", 1))
            percent = freshness_percent(last_updated, period_s)
            next_update = estimate_next_update(last_updated, period_s, True)

        result = build_status(
            name=self.section,
            kind="layer",
            percent=percent,
            last_updated=last_updated,
            enabled=self.enabled,
            next_update=next_update,
            detail=detail,
            status=status,
        )
        result["segments"] = segments
        result["run_date"] = run_date
        result["run_id"] = run_id
        return result

    def _now_fhour(self, run_date, run_id) -> int:
        """The forecast hour of `run_date`/`run_id` whose valid time is closest to
        wall-clock now — the same "nearest hour" the frontend scrubber's nowHour()
        computes, so layer_status()'s now-onward filtering matches what the scrubber
        will actually let a user reach."""
        epoch = resolve_run_epoch_utc(run_date, run_id)
        now = datetime.now(timezone.utc)
        return max(0, int(round((now - epoch).total_seconds() / 3600.0)))

    def latest_store_run(self, products):
        """Resolve the freshest run actually present in the fieldstore catalog for the
        given products, returning (run_date, run_id, hours) or None.

        Field-reading layers should resolve their run from the CATALOG, not from the
        cached GFS/RTOFS baseline. The baseline tracks what NOMADS has *published*, which
        can run ahead of what the collector has *ingested* — and, because the baseline is
        cached per process, it can also drift behind once it goes stale. Reading the
        catalog renders exactly what is on disk. Scope by `products` so independent model
        cycles (GFS 00/06/12/18 vs RTOFS "00") resolve to their own run.
        """
        try:
            store = self._store
            avail = store.field_catalog_adapter.get_latest_run_hours(products=list(products))
        except Exception as e:
            logger.warning(f"{self.section}: catalog run lookup failed: {e}")
            return None
        if not avail or not avail.get("hours"):
            return None
        return avail["run_date"], avail["run_id"], avail["hours"]

    def _resolve_forecast_state(
        self, *, baseline_key: str, resolve_fn, label: str
    ) -> "ForecastState":
        """Shared baseline-cache-or-fetch + forecast-hour math backing get_gfs_state()/
        get_rtofs_state() (~75% identical before this extraction, differing only in
        which baseline they cache/resolve and their log labels). The first updater to
        need `baseline_key` this cycle resolves it (a network sync via `resolve_fn`);
        every other updater this cycle reads the cached result from
        map_data.shared_state. Returns the resolved ForecastState (does not mutate
        self)."""
        baseline = getattr(self.map_data, "shared_state", {}).get(baseline_key)

        # ESTABLISH THE DATUM (only runs once per map refresh)
        if not baseline:
            logger.debug(f"Section {self.section} setting up {label} baseline")
            baseline = resolve_fn()
            if not baseline:
                raise RuntimeError(f"Failed to sync {label} baseline from NOMADS.")
            self.map_data.shared_state[baseline_key] = baseline
            logger.debug(
                f"{label} Baseline Synced: {baseline['date_str']} {baseline['run']}Z"
            )

        # CALCULATE THE DYNAMIC OFFSET (runs for every layer)
        now = datetime.now(timezone.utc)
        user_offset_hours = self.forecast_hour
        hours_since_run = int(
            round((now - baseline["timestamp"]).total_seconds() / 3600.0)
        )
        true_f_hour = max(0, hours_since_run + user_offset_hours)

        state = ForecastState.at_hour(baseline["date_str"], baseline["run"], true_f_hour)
        logger.debug(
            f"Section {self.section} get_{label.lower()}_state: forecast hour "
            f"{state.forecast_hour_str}; date_str {state.run_date_str}; run {state.run_id}"
        )
        return state

    def get_gfs_state(self) -> "ForecastState":
        """
        Lazy evaluation: The first updater to call this method performs a quick network
        sync to establish the GFS datum. All subsequent updaters read from memory.
        """
        from atmos_gl.lib.gfs import resolve_gfs_baseline

        return self._resolve_forecast_state(
            baseline_key="gfs_baseline", resolve_fn=resolve_gfs_baseline, label="GFS"
        )

    def get_rtofs_state(self) -> "ForecastState":
        """RTOFS (ocean) analogue of get_gfs_state, for currents and future ocean
        layers. Resolves the daily RTOFS run (its own cycle, cached separately in
        shared_state) and returns the SAME ForecastState shape get_gfs_state does, so
        render_all_hours and the fieldstore reads work unchanged — they simply operate
        on the RTOFS run.

        RTOFS is one 00Z cycle/day; 'now' is hours-since-analysis, and a per-layer
        forecast_hour offset steps forward, identical in spirit to the GFS path.
        """
        from atmos_gl.lib.rtofs import resolve_rtofs_baseline

        return self._resolve_forecast_state(
            baseline_key="rtofs_baseline", resolve_fn=resolve_rtofs_baseline, label="RTOFS"
        )


class MultiHourRenderMixin:
    """Per-forecast-hour render-caching machinery, mixed into Updater subclasses that
    render one output per forecast hour (isobars, wind, precipitation, currents, waves,
    and the scalar-field trio via ScalarFieldUpdater) rather than once per cycle.

    Single-shot layers (sst, clouds, markers) never mix this in — they render once per
    cycle, not per forecast hour, so should_plot_for_hour's per-hour freshness check and
    render_all_hours' gap-filling loop don't apply to them (architecture review
    candidate "slim Updater" — these 4 methods used to sit on Updater itself, inherited
    by every layer including ones that could never call them).

    Assumes it's mixed into an Updater subclass: uses self.output_path, self._store,
    self.per_hour_outputs, self.process_status_adapter, self.section,
    self.latest_store_run() and self.get_db_field_at_hour() (the last two stay on
    Updater itself, since markers.py — a single-shot layer — also calls
    get_db_field_at_hour directly, to sample weather at a specific hour rather than to
    render one). Updater.layer_status()'s multi-hour branch also calls
    should_plot_for_hour, but only when self.status_product is set — which only
    multi-hour subclasses do, and they always pair it with this mixin. Which forecast
    run + hour a call operates on is always passed explicitly as a ForecastState — see
    that class's docstring and CONTEXT.md's "ForecastState" entry.
    """

    def get_output_path_for_hour(self, fhour: int | str) -> str:
        """Return a per-hour output path for caching renders.

        The path is:
          {base_path}_f{fhour:03d}.png

        Example: "/path/to/precipitation_f003.png"
        """
        fhour = int(fhour)

        base, ext = os.path.splitext(self.output_path)
        return f"{base}_f{fhour:03d}{ext}"

    def publish_current_hour(self, fhour: int | str):
        """Publish the given forecast hour's render to the STABLE base filename.

        The backend caches per-hour ({base}_fNNN.png and {base}_fNNN_data.png), but the
        frontend fetches the run-agnostic base names ({base}.png and {base}_data.png) —
        it has no way to know which forecast hour is valid "now". This copies the
        per-hour outputs to those base names so the frontend always sees the latest hour.

        Copies whichever of the two artifacts exist (static PNG and/or _data.png texture),
        using atomic replace so the frontend never reads a half-written file.
        """
        fhour = int(fhour)

        base, ext = os.path.splitext(self.output_path)
        per_hour = f"{base}_f{fhour:03d}{ext}"

        pairs = [
            (per_hour, self.output_path),  # static raster
            (
                f"{base}_f{fhour:03d}_data.png",
                f"{base}_data.png",
            ),  # multi-frame texture
        ]
        import shutil

        for src, dst in pairs:
            if not os.path.exists(src):
                continue
            try:
                tmp = f"{dst}.tmp"
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                logger.debug(
                    f"{self.section}: published {os.path.basename(src)} -> {os.path.basename(dst)}"
                )
            except Exception as e:
                logger.warning(f"{self.section}: failed to publish {src} -> {dst}: {e}")

    def should_plot_for_hour(
        self, state: "ForecastState", product_name: str, settings_sig: str | None = None
    ) -> bool:
        """Check if a per-hour output needs updating.

        Returns True if:
          - The output file doesn't exist, OR
          - settings_sig is given and doesn't match the '<output>.sig' sidecar, OR
          - The field's valid_time is newer than the output file's mtime

        Returns False if the file is already fresh. This prevents re-plotting
        when data hasn't changed. Uses the catalog metadata only (no array load).

        settings_sig (optional): a string from Updater._settings_signature(),
        capturing the render-relevant settings (e.g. isobars' linewidth/color/step).
        A settings-only change touches neither the output file's mtime nor the data's
        updated_at, so without this check an edited setting would never actually
        re-render an already-cached hour -- mirrors Updater._is_render_fresh's
        identical '<out>.sig' sidecar, used by single-shot layers. Missing/mismatched
        sig counts as stale. Callers that don't pass one (None, the default) keep the
        old data-only freshness behaviour unchanged.
        """
        output_path = self.get_output_path_for_hour(state.fhour)
        base, ext = os.path.splitext(output_path)

        # A complete render produces every suffix in self.per_hour_outputs. If ANY is
        # missing, re-plot to fill the gap (this is what makes "delete a _data.png to
        # force regeneration" work even when the static .png is still present).
        required_paths = []
        for suffix in self.per_hour_outputs or [".png"]:
            # ".png" -> the static per-hour file; "_data.png"/"_labels.geojson" -> base+suffix.
            required_paths.append(output_path if suffix == ext else f"{base}{suffix}")
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            return True

        if settings_sig is not None:
            try:
                with open(f"{output_path}.sig") as f:
                    if f.read() != settings_sig:
                        return True
            except FileNotFoundError:
                return True

        # All outputs exist — check freshness against when the data was written.
        # Use the static PNG's mtime as the reference (oldest-equivalent; all outputs
        # are written together in one plot() call).
        try:
            fs = self._store
            meta = fs.get_field_meta(
                state.run_date_str, state.run_id, state.fhour, product_name
            )

            if not meta or meta.get("updated_at") is None:
                # No data catalogued, don't plot (data isn't ready yet)
                return False

            # Get file's mtime and compare to when the DATA ROW was last written.
            # NOTE: use updated_at (when the field was stored), NOT valid_time (the
            # forecast's validity time, which is usually in the future and would make
            # every hour look "newer" than its PNG, forcing a re-plot every cycle).
            file_mtime = min(os.path.getmtime(p) for p in required_paths)
            file_dt = datetime.fromtimestamp(file_mtime, tz=timezone.utc)

            field_updated = meta.get("updated_at")
            if field_updated is None:
                return False

            # Ensure both are tz-aware for comparison
            if field_updated.tzinfo is None:
                field_updated = field_updated.replace(tzinfo=timezone.utc)

            # Plot if data is newer (with a 1-second tolerance for clock skew)
            return (field_updated - file_dt).total_seconds() > 1

        except Exception as e:
            logger.debug(
                f"should_plot_for_hour({product_name}, f{state.fhour:03d}) check failed: {e}"
            )
            # On error, be conservative — don't plot (file is probably fine)
            return False

    def render_all_hours(
        self, product_name, plot_fn, field_ready, max_hours=None, settings_sig=None
    ):
        """Gap-filling per-hour render loop.

        The scrubber needs a rendered PNG for every forecast hour that has data, not
        just the current one. This loops over the hours present in the catalog for
        this run and plots any whose output is missing or stale (should_plot_for_hour
        decides per hour). Hours already rendered and fresh are skipped cheaply, so
        steady state is N metadata checks and zero re-renders; only newly-arrived or
        deleted hours actually plot.

        Args:
            product_name: catalog product key (e.g. "isobars").
            plot_fn: callable(field, state) that renders + writes the per-hour outputs
                     for the given ForecastState.
            field_ready: callable(field) -> bool; whether the fetched field has the
                     data this layer needs (e.g. values is not None; u/v for wind).
            max_hours: stop after actually plotting this many hours (None = drain the
                     whole backlog in one call, the original behaviour). layer_builder
                     passes 1 so one process-pool dispatch renders one hour and yields
                     the worker back to the round-robin queue, instead of one layer
                     monopolising a worker until its entire backlog is caught up
                     (architecture review candidate "interleave per-hour rendering
                     across layers").
            settings_sig: optional signature (Updater._settings_signature()) of this
                     layer's render-relevant settings, forwarded to should_plot_for_hour
                     so a settings-only change (e.g. isobars' linewidth) invalidates
                     already-cached hours the same way a data change does. None (the
                     default) keeps the old data-only freshness behaviour.

        A "hour actually (re)plotted" means plot_fn ran without raising AND
        should_plot_for_hour now reports the hour complete -- not merely that plot_fn
        returned. Some plot_fn implementations (WindUpdater, ScalarFieldUpdater)
        deliberately swallow a partial internal failure so it doesn't also block an
        independent output (see issue #283); without this recheck, a hour that fails
        the same way on every attempt would count as "done" here, get re-selected as
        the earliest still-pending hour on every future call, and starve every later
        hour behind it forever under max_hours=1's round-robin dispatch.

        Returns the number of hours actually (re)plotted.
        """
        # Resolve the run from the CATALOG (what's actually ingested), not from a
        # baseline-derived state (which can be stale or ahead of the collector). Each
        # hour gets its own ForecastState, built fresh -- no instance state to save or
        # restore, so callers that resolve their own baseline state beforehand (e.g.
        # the waves heat-tile GRIB download) are unaffected by construction.
        resolved = self.latest_store_run([product_name])
        if not resolved:
            logger.info(
                f"{self.section}: no hours in catalog yet (collector may not have run)."
            )
            return 0
        run_date, run_id, hours = resolved

        # Hours before "now" are still retained in the catalog (data_collector.
        # cache_hours is wider than what's reachable) but the scrubber's minHour only
        # ever ratchets forward -- see layer_status()'s identical filter and its
        # docstring. Applying the same filter here (not just in status reporting)
        # matters most right after a section re-enables from a long-off channel: with
        # max_hours=1 per round-robin turn, spending turns re-rendering unreachable
        # past hours first delays reaching the one range a user can actually see.
        now_fhour = self._now_fhour(run_date, run_id)
        hours = [h for h in hours if h >= now_fhour]

        # Only pass settings_sig through when the caller actually gave one, so callers
        # (and tests) that don't care about settings-driven staleness keep calling
        # should_plot_for_hour with its original two-argument shape.
        sig_kwargs = {"settings_sig": settings_sig} if settings_sig is not None else {}

        plotted = 0
        attempted = 0
        examined = 0
        for fh in hours:
            examined += 1
            state = ForecastState.at_hour(run_date, run_id, fh)
            if not self.should_plot_for_hour(state, product_name, **sig_kwargs):
                continue
            field = self.get_db_field_at_hour(state, product_name)
            if not field or not field_ready(field):
                continue
            attempted += 1
            try:
                plot_fn(field, state)
                # Advance last_updated as each hour lands, not just once the whole
                # cycle (every TASK_CLASSES entry) finishes — a multi-hour layer can
                # take a long time to catch up on a cold start, and the Data Status
                # UI's percent bar already reflects per-hour progress live; last_updated
                # should too instead of sitting on "never" for the whole cycle.
                self.process_status_adapter.record_process_run(self.section, "layer", success=True)
                # Publish THIS hour to the stable base filename immediately, not once
                # at the end -- with max_hours capping each call to one hour, "once at
                # the end" would mean "never" until the whole backlog drains. The
                # tradeoff: while catching up a multi-hour backlog, the stable file can
                # briefly point at an older hour than it did a moment ago (hours render
                # in ascending order) before reaching the true latest again -- accepted
                # in exchange for every layer visibly progressing instead of one at a
                # time. Published even when the hour turns out still-incomplete below
                # (e.g. a swallowed partial failure) -- whatever DID get written should
                # still reach the frontend's stable filename right away.
                self.publish_current_hour(state.fhour)
                # Written BEFORE the should_plot_for_hour recheck below, so a genuinely
                # complete render is recognised as such by that same call rather than
                # reading its own just-written signature as still "missing".
                if settings_sig is not None:
                    self._write_render_signature(
                        self.get_output_path_for_hour(state.fhour), settings_sig
                    )
                # plot_fn not raising is NOT proof this hour is done: WindUpdater and
                # ScalarFieldUpdater deliberately swallow a known, deterministic Cartopy
                # antimeridian bug (issue #283/PR #281) internally so a contourf failure
                # doesn't also block their texture output. That means the same hour can
                # fail this way on every attempt and never produce a complete
                # per_hour_outputs set. Crediting that as a "success" toward max_hours
                # would make this method re-select the same permanently-broken hour
                # (always the earliest pending one, since hours are ascending) on every
                # future call forever -- under max_hours=1's round-robin dispatch that
                # starves every later hour behind it (observed live: Data Status stuck
                # at 0% while the log repeats "static render fNNN failed" for one hour).
                # Re-checking here is cheap (catalog metadata only, no array load) and
                # only credits genuinely-complete hours, so the loop moves on to try a
                # later hour instead of stalling on this one.
                if not self.should_plot_for_hour(state, product_name, **sig_kwargs):
                    plotted += 1
            except Exception as e:
                logger.warning(f"{self.section}: plot f{state.fhour:03d} failed: {e}")
            if max_hours is not None and plotted >= max_hours:
                break

        stopped_early = examined < len(hours)
        if plotted:
            suffix = (
                f"({len(hours)} available, stopped early after {examined} examined)"
                if stopped_early
                else f"({len(hours)} available, {len(hours) - plotted} already fresh)"
            )
            logger.info(f"{self.section}: rendered {plotted} hour(s) {suffix}.")
        elif attempted:
            logger.warning(
                f"{self.section}: attempted {attempted} hour(s) but none completed "
                f"({len(hours)} available) -- still incomplete, will retry next cycle."
            )
        else:
            logger.debug(
                f"{self.section}: all {len(hours)} hour(s) fresh; nothing to render."
            )
        return plotted
