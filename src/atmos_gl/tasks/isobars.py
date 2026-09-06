#!/usr/bin/env python3
import os
import json
import logging
import numpy as np
import matplotlib.patheffects as patheffects
import cartopy.crs as ccrs


# Internal imports
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.texture import encode_frames
from .common import Updater, MapData, MultiHourRenderMixin, ForecastState
from .plotting import Plot, clamp_lats_to_mercator_limit

logging.getLogger("gribapi.bindings").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class IsobarUpdater(Updater, MultiHourRenderMixin):
    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "Isobars", map_data)
        # Physical bounds for the shader encoding (must match the frontend).
        # 950hPa (severe cyclone) to 1050hPa (strong anticyclone).
        self.VMIN_PRESSURE = 950.0
        self.VMAX_PRESSURE = 1050.0
        # Static PNG + GPU data texture + vector pressure labels.
        self.per_hour_outputs = [".png", "_data.png", "_labels.geojson"]
        self.status_product = "isobars"

    def plot(self, field0, state: ForecastState):
        """Render the static isobar PNG (from frame 0) AND the N-frame data texture.

        Now consumes pre-processed fields from the DB.
        Outputs are cached per-hour.
        """
        logger.debug("Plotting isobars to per-hour output path")

        lats = clamp_lats_to_mercator_limit(field0["lat"])
        lons = field0["lon"]
        p = field0["values"]  # already smoothed from unpacker
        # Isobars render straight from the native grid (no regrid_for_lod call here),
        # so the antimeridian seam has to be closed directly -- see
        # close_lon_seam_for_contour's docstring for why contour() needs this at all.
        lons, p = self.close_lon_seam_for_contour(lons, p)

        plot = Plot(self.map_data.region)
        plot.get_figure()

        step = self.settings.get("isobar_step", 4)
        levels = np.arange(940, 1060, step)
        color = self.settings.get("isobar_color", "white")
        f_size = self.settings.get("label_fontsize", 10)
        thickness = self.settings.get("linewidth", 1.0)
        alpha_val = float(self.settings.get("opacity", 100) / 100)

        line_effect = [
            patheffects.withStroke(linewidth=thickness + 2, foreground="black")
        ]

        plot.ax.contour(
            lons,
            lats,
            p,
            levels=levels,
            colors=color,
            linewidths=thickness,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # Add labels at isosurfaces. Label EVERY level; matplotlib's clabel handles
        # spacing/placement along each contour. We harvest these label positions
        # (lon/lat/value) into a per-hour GeoJSON for the frontend's vector text
        # layer — the GPU line layer can't render numbers itself.
        cs = plot.ax.contour(
            lons,
            lats,
            p,
            levels=levels,
            colors=color,
            linewidths=thickness,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        label_artists = plot.ax.clabel(
            cs,
            inline=True,
            fontsize=f_size,
            fmt="%1.0f",
            colors=color,
            manual=False,
        )

        for text in plot.ax.texts:
            text.set_path_effects(line_effect)
            text.set_alpha(alpha_val)

        # Per-hour output path
        output_path_for_hour = self.get_output_path_for_hour(state.fhour)
        plot.save_figure(output_path_for_hour)

        # Harvest label positions BEFORE closing the figure.
        base, _ = os.path.splitext(output_path_for_hour)
        self._write_labels_geojson(label_artists, plot.ax, f"{base}_labels.geojson")

        plt_close = getattr(plot, "close", None)
        if callable(plt_close):
            plt_close()

        # --- WebGL single-hour data texture (one frame per forecast hour;
        # the frontend scrubber assembles the animation from consecutive hours) ---
        encode_frames(
            [field0["values"]],
            f"{base}_data.png",
            self.VMIN_PRESSURE,
            self.VMAX_PRESSURE,
        )
        logger.info(f"Finished Isobars texture f{state.fhour:03d}.")

    def _write_labels_geojson(self, label_artists, ax, out_path):
        """Write contour-label positions as a GeoJSON FeatureCollection of points.

        Each feature: Point [lon, lat] with properties {label: "1012"}.

        IMPORTANT: clabel Text artists report their position in the AXES projection
        (here Web Mercator metres), NOT lon/lat. We must transform each position back
        to PlateCarree (lon/lat) via the axes CRS, or every label fails the lat range
        check and the file ends up empty.
        Written atomically (tmp + replace) so the frontend never reads a half file.
        """
        geo = ccrs.PlateCarree()
        proj = getattr(ax, "projection", geo)  # the axes CRS (e.g. Mercator.GOOGLE)
        features = []
        for t in label_artists or []:
            try:
                x, y = t.get_position()  # axes-projection coords (metres)
                label = t.get_text().strip()
                if not label:
                    continue
                lon, lat = geo.transform_point(float(x), float(y), proj)  # -> lon/lat
                if not (np.isfinite(lon) and np.isfinite(lat)):
                    continue
                if not (-90.0 <= lat <= 90.0):
                    continue
                lon = ((lon + 180.0) % 360.0) - 180.0  # normalise to [-180,180)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [round(lon, 3), round(lat, 3)],
                        },
                        "properties": {"label": label},
                    }
                )
            except Exception:
                continue

        fc = {"type": "FeatureCollection", "features": features}
        try:
            tmp = f"{out_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(fc, f, separators=(",", ":"))
            os.replace(tmp, out_path)
            logger.debug(
                f"Wrote {len(features)} isobar labels -> {os.path.basename(out_path)}"
            )
        except Exception as e:
            logger.warning(f"Failed to write isobar labels {out_path}: {e}")

    def _render_settings_signature(self) -> str:
        """Render-relevant settings for should_plot_for_hour's staleness check
        (mirrors SSTUpdater._mode_settings_signature). Isobars bakes all of these
        directly into the per-hour PNG/labels in plot() above, so a change to any of
        them must invalidate an already-cached hour even though neither the source
        data nor the output file's mtime changed on its own."""
        return self._settings_signature(
            {
                "isobar_step": self.settings.get("isobar_step", 4),
                "isobar_color": self.settings.get("isobar_color", "white"),
                "linewidth": self.settings.get("linewidth", 1.0),
                "label_fontsize": self.settings.get("label_fontsize", 10),
                "opacity": self.settings.get("opacity", 100),
            }
        )

    def run(self, max_hours=None):
        # Warms the shared per-cycle GFS baseline cache (map_data.shared_state) for
        # other updaters this cycle; render_all_hours resolves its own state from the
        # catalog below, so the return value here is unused.
        self.get_gfs_state()
        # Render EVERY available forecast hour (gap-filling), so the scrubber has
        # a PNG for each hour. should_plot_for_hour skips hours already fresh.
        # max_hours=1 from layer_builder's round-robin dispatch renders one hour and
        # returns, so this layer doesn't monopolise a render-pool worker.
        return self.render_all_hours(
            "isobars",
            plot_fn=self.plot,
            field_ready=lambda f: f.get("values") is not None,
            max_hours=max_hours,
            settings_sig=self._render_settings_signature(),
        )
