#!/usr/bin/env python3
import os
import math
import logging

import numpy as np
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.texture import encode_uv
from .common import Updater, MapData, MultiHourRenderMixin, ForecastState
from .plotting import Plot, clamp_lats_to_mercator_limit

logging.getLogger("cfgrib").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# windy.com-style wind-speed ramp (calm -> storm). Kept in sync with the frontend
# PALETTE in ui/modules/wind.js so the matplotlib static heatmap and the GPU per-hour
# heatmap (which shades the velocity texture) look identical.
WIND_PALETTE = [
    (0.25, 0.30, 0.60),   # calm   - deep blue
    (0.15, 0.60, 0.85),   # light  - cyan-blue
    (0.20, 0.75, 0.45),   # breeze - green
    (0.95, 0.90, 0.30),   # fresh  - yellow
    (0.95, 0.55, 0.20),   # strong - orange
    (0.90, 0.20, 0.20),   # gale   - red
    (0.75, 0.25, 0.85),   # storm  - violet
]
WIND_CMAP = mcolors.LinearSegmentedColormap.from_list("windy_wind", WIND_PALETTE)


class WindUpdater(Updater, MultiHourRenderMixin):
    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "Wind", map_data)
        self.VMAX_WIND = 40.0          # m/s encoding range for the velocity texture
        self.level_of_detail = int(self.settings.get("level_of_detail", 1))
        self.lod_desc = None
        # Heatmap speed scale — computed from actual data in run() (global max across all
        # hours, rounded up to the nearest 10 km/h). Initialised to a safe default so
        # plot() can be called standalone without a prior run().
        self.VMAX_SPEED = 100.0 / 3.6   # m/s; overwritten by run()
        # Per hour: windspeed heatmap (.png, like temperature) + velocity texture
        # (_data.png, decoded as u,v by the particle shader AND shaded as speed by the
        # frontend GPU heatmap).
        self.per_hour_outputs = [".png", "_data.png"]
        self.status_product = "wind"

    def plot(self, field0, state: ForecastState):
        """Render the per-hour windspeed heatmap (.png) + velocity texture (_data.png).

        The particle shader decodes _data.png's rg as (u, v) via `rg * (2*vmax) - vmax`;
        the frontend heatmap re-uses the same texture, computing speed = |(u, v)|. The
        .png is a matplotlib heatmap for the non-stepping (static) view.
        """
        u = field0["u"]  # m/s
        v = field0["v"]  # m/s
        lats = field0["lat"]
        lons = field0["lon"]
        speed = np.hypot(u, v)

        # --- windspeed heatmap (mirrors TemperatureUpdater.plot) ---
        new_lats, new_lons, spd_smooth = self.regrid_for_lod(speed, lats, lons)

        out_for_hour = self.get_output_path_for_hour(state.fhour)

        # The static heatmap (contourf) and the velocity texture (encode_uv, below)
        # are independent outputs -- a contourf failure must not block the texture,
        # since that's what the frontend's animated WebGL wind layer actually reads.
        # contourf occasionally hits a known Cartopy bug (antimeridian-wrapping
        # polygon reprojection inside its own get_datalim() call) for specific,
        # deterministic field topologies -- see ScalarFieldUpdater.plot() (PR #281)
        # for the same fix applied there; issue #283 tracks this one.
        try:
            plot = Plot(self.map_data.region)
            plot.get_figure()
            norm = mcolors.Normalize(vmin=0.0, vmax=self.VMAX_SPEED)
            # Closes the antimeridian seam for this static render only -- see
            # close_lon_seam_for_contour's docstring. regrid_for_lod's own new_lons/
            # spd_smooth stay untouched for any other use (there is none here: the
            # velocity texture below is encoded via encode_uv from field0's native
            # u/v grids, not this LOD-regridded one).
            contour_lons, contour_spd = self.close_lon_seam_for_contour(new_lons, spd_smooth)
            plot.ax.contourf(
                contour_lons,
                clamp_lats_to_mercator_limit(new_lats),
                contour_spd,
                levels=20,
                cmap=WIND_CMAP,
                norm=norm,
                transform=ccrs.PlateCarree(),
                extend="max",
                zorder=2,
            )
            plot.save_figure(out_for_hour)

            plt_close = getattr(plot, "close", None)
            if callable(plt_close):
                plt_close()
        except Exception as e:
            logger.warning(
                f"{self.section}: static render f{state.fhour:03d} failed: {e}"
            )

        # --- velocity texture: raw field; frontend applies direction-coherence live ---
        base, _ = os.path.splitext(out_for_hour)
        encode_uv(u, v, f"{base}_data.png", self.VMAX_WIND, lat=field0.get("lat"))
        logger.info(
            f"Finished Wind f{state.fhour:03d} (heatmap .png + R=U,G=V texture)."
        )

    def run(self, max_hours=None):
        # Warms the shared per-cycle GFS baseline cache (map_data.shared_state) for
        # other updaters this cycle; render_all_hours resolves its own state from the
        # catalog below, so the return value here is unused.
        self.get_gfs_state()

        # --- pre-scan: find global max wind speed across all available hours ----------
        # We want a SINGLE vmax for the whole run so the palette means the same speed
        # at every hour (temporal blending between hours with different scales is wrong).
        # Round up to the nearest 10 km/h so the legend has clean tick values.
        # Resolve the run from the catalog (what's ingested) so the scan and the
        # subsequent render_all_hours agree on exactly the same run + hours.
        resolved = self.latest_store_run(["wind"])
        if resolved:
            run_date, run_id, hours = resolved
        else:
            run_date, run_id, hours = None, None, []

        max_speed_ms = 0.0
        for fh in (hours or []):
            state = ForecastState.at_hour(run_date, run_id, fh)
            field = self.get_db_field_at_hour(state, "wind")
            if field and field.get("u") is not None and field.get("v") is not None:
                peak = float(np.hypot(field["u"], field["v"]).max())
                if peak > max_speed_ms:
                    max_speed_ms = peak

        if max_speed_ms > 0:
            max_kph = max_speed_ms * 3.6
            rounded_kph = math.ceil(max_kph / 10) * 10   # e.g. 51.7 -> 60
            self.VMAX_SPEED = rounded_kph / 3.6
            logger.info(
                f"Wind: heatmap scale = {rounded_kph} km/h "
                f"(data peak {max_kph:.1f} km/h across {len(hours)} hours)"
            )
        else:
            self.VMAX_SPEED = 100.0 / 3.6
            logger.info("Wind: no field data for pre-scan; using 100 km/h default scale")

        # --- write meta for the frontend (fetched by wind.js to set the shader scale) --
        if self.output_path:
            import json
            meta_path = os.path.join(
                os.path.dirname(self.output_path), "wind_meta.json"
            )
            try:
                with open(meta_path, "w") as f:
                    json.dump({"heatmap_max_kph": round(self.VMAX_SPEED * 3.6)}, f)
            except Exception as e:
                logger.warning(f"Wind: could not write wind_meta.json: {e}")

        # --- render all hours (plot() now has VMAX_SPEED set globally) ----------------
        # max_hours=1 from layer_builder's round-robin dispatch renders one hour and
        # returns, so this layer doesn't monopolise a render-pool worker -- the pre-scan
        # above re-runs every such call (cheap DB reads, not full renders, so acceptable).
        return self.render_all_hours(
            "wind",
            plot_fn=self.plot,
            field_ready=lambda f: f.get("u") is not None and f.get("v") is not None,
            max_hours=max_hours,
        )