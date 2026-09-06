#!/usr/bin/env python3
"""One renderer for every single-scalar contourf heatmap, driven by a small spec.

temperature, ozone, stormwatch (CAPE) and pwat render identically — regional LOD
regrid, a 20-level `contourf`, a per-hour PNG, a hour-independent colourbar key, and a
WebGL data texture — differing only in colour range/mode and a couple of specs' extra
"critical zone" rendering (see below). That variation is captured in `ScalarFieldSpec`;
`SPECS` holds the four instances and `ScalarFieldUpdater` consumes one. Adding a plain
(unthresholded) fifth scalar field is a single `SPECS` entry plus one `TASK_CLASSES`
line in layer_builder.

ozone and pwat use the alternative threshold-based "critical zone" rendering instead of
a plain named `cmap` -- see `ScalarFieldSpec`'s threshold_*/focus/palette_* fields and
`_threshold_colormap()`. This restores behaviour PR #49 silently dropped: ozone used to
have its own bespoke `OzoneUpdater` with a "critical" palette highlighting ozone holes
below a configurable threshold, lost when it was collapsed into this shared renderer
(the settings UI's "Critical Ozone Threshold" slider was left wired to nothing). pwat
reuses the same mechanism, mirrored (its "problem zone" is above the threshold, not
below).

Validated with ast.parse.
"""
import os
import logging
from dataclasses import dataclass

import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.texture import encode_frames
from .common import MapData, ForecastState
from .plotting import Plot, clamp_lats_to_mercator_limit
from .single_hour_scalar import SingleHourScalarUpdater

logging.getLogger("cfgrib").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalarFieldSpec:
    """Everything that varies between the scalar-field heatmaps.

    `product` is the catalog/status key, passed to render_all_hours, reported as
    status_product, and used as the Updater section name for settings lookup (Updater
    lowercases the section, so one identifier serves all three roles).

    `cmap` drives the plain (unthresholded) render used by `temperature`/`stormwatch`.
    The threshold_* / focus / palette_* fields below are the alternative "critical
    zone" rendering `ozone` and `pwat` use instead — see `_threshold_colormap()`.
    Leave all of them None (the dataclass default) for a plain-cmap spec; setting
    `threshold_setting` is what switches `ScalarFieldUpdater.plot()` onto that path.
    """

    product: str
    vmin: float
    vmax: float
    extend: str
    ticks: list
    title: str
    cmap: str | None = None

    # --- threshold ("critical zone") rendering -- ozone, pwat ---
    # Settings key holding the live threshold value (e.g. "critical_du"), and its
    # fallback default if the setting is absent.
    threshold_setting: str | None = None
    threshold_default: float | None = None
    # Which side of the threshold gets the graded palette colour; the other side
    # renders as `flat_color` (fixed, not graded) -- "below" (ozone: the hole gets
    # worse toward vmin) or "above" (pwat: moisture gets worse toward vmax).
    focus: str | None = None
    flat_color: tuple | None = None
    # Settings key selecting a named entry from `palettes` (e.g. "palette"), its
    # fallback name, and the {name: [colour, ...]} registry itself. Colours are
    # evenly spaced across the graded side, first colour at the threshold boundary,
    # last colour at the domain's extreme edge (a 2-colour list is a simple fade; a
    # longer list, e.g. pwat's "standard", reuses a full multi-stop ramp).
    palette_setting: str | None = None
    palette_default: str | None = None
    palettes: dict | None = None


def _threshold_colormap(vmin, vmax, threshold, focus, palette_colors, flat_color):
    """Build a colormap spanning [vmin, vmax]: one side of `threshold` grades through
    `palette_colors` (first colour at the threshold boundary, last at the domain's
    extreme edge), the other side is flat `flat_color`. `focus="below"` grades toward
    vmin (ozone: worse toward the lowest reading); `focus="above"` grades toward vmax
    (pwat: worse toward the highest reading). A small transition band softens the seam
    between the graded and flat sides instead of a hard cut.
    """
    span = max(1e-9, vmax - vmin)
    t = max(0.0, min(1.0, (threshold - vmin) / span))
    band = 0.01
    n = len(palette_colors)
    extreme_edge = 0.0 if focus == "below" else 1.0

    def pos_at(i):
        return t if n == 1 else t + (i / (n - 1)) * (extreme_edge - t)

    stops = [(pos_at(i), c) for i, c in enumerate(palette_colors)]
    if focus == "below":
        stops.append((min(1.0, t + band), flat_color))
        stops.append((1.0, flat_color))
    else:
        stops.append((0.0, flat_color))
        stops.append((max(0.0, t - band), flat_color))

    # LinearSegmentedColormap requires strictly increasing positions.
    deduped = []
    for pos, c in sorted(stops, key=lambda s: s[0]):
        if deduped and pos <= deduped[-1][0]:
            pos = deduped[-1][0] + 1e-6
        deduped.append((pos, c))
    return mcolors.LinearSegmentedColormap.from_list("threshold_mask", deduped, N=256)


# Ozone's original "critical" palette (a hand-built magenta/yellow gradient marking
# ozone holes) -- restored here after PR #49 collapsed the bespoke OzoneUpdater into
# this shared renderer and silently dropped it, leaving the settings UI's
# "Critical Ozone Threshold" slider wired to nothing. Brightest colour (yellow) now
# marks the most severe reading, consistent across both specs below.
_OZONE_PALETTES = {
    "alert": [(1.0, 0.0, 1.0), (1.0, 1.0, 0.0)],  # magenta (threshold) -> yellow (worst)
    "high_contrast": [(1.0, 0.0, 0.0), (1.0, 1.0, 0.8)],  # red (threshold) -> pale yellow (worst)
}
_OZONE_FLAT = (0.0, 0.1, 0.3, 0.2)  # dim, mostly-transparent -- the "safe" zone

# pwat's palettes -- "standard" reuses precipitation.py's exact 7-stop ramp so the two
# layers visually reinforce each other when both render at once; the other two are
# NOAA's actual blue/purple convention for rendering moisture plumes, kept distinct
# from precipitation's warm colours as a second, unambiguous option.
_PWAT_PALETTES = {
    "standard": [
        (0.0, 1.0, 1.0), (0.0, 0.5, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
        (1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
    ],
    "atmospheric_river": [(0.0, 0.0, 0.55), (0.6, 0.0, 0.85)],  # deep blue -> violet
    "deep_teal": [(0.7, 1.0, 1.0), (0.0, 0.35, 0.3)],  # pale cyan -> deep teal
}
_PWAT_FLAT = (0.0, 0.0, 0.0, 0.0)  # fully transparent -- unremarkable moisture

SPECS = {
    "temperature": ScalarFieldSpec(
        product="temperature",
        cmap="RdYlBu_r",
        vmin=-40.0,
        vmax=50.0,
        extend="both",
        ticks=[-40, -20, 0, 10, 20, 30, 40, 50],
        title="Temperature (°C)",
    ),
    "ozone": ScalarFieldSpec(
        product="ozone",
        vmin=150.0,
        vmax=500.0,
        extend="neither",
        ticks=[150, 200, 250, 300, 350, 400, 450, 500],
        title="Ozone (DU)",
        threshold_setting="critical_du",
        threshold_default=220.0,
        focus="below",
        flat_color=_OZONE_FLAT,
        palette_setting="palette",
        palette_default="alert",
        palettes=_OZONE_PALETTES,
    ),
    "stormwatch": ScalarFieldSpec(
        product="stormwatch",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=5000.0,
        extend="max",
        ticks=[0, 1000, 2000, 3000, 4000, 5000],
        title="CAPE (J/kg)",
    ),
    "pwat": ScalarFieldSpec(
        product="pwat",
        vmin=0.0,
        vmax=80.0,
        extend="neither",
        ticks=[0, 20, 40, 50, 60, 80],
        title="Precipitable Water (mm)",
        threshold_setting="critical_pwat",
        threshold_default=50.0,
        focus="above",
        flat_color=_PWAT_FLAT,
        palette_setting="palette",
        palette_default="standard",
        palettes=_PWAT_PALETTES,
    ),
}


class ScalarFieldUpdater(SingleHourScalarUpdater):
    def __init__(self, config: AtmosGLConfig, map_data: MapData, spec: ScalarFieldSpec):
        super().__init__(config, spec.product, map_data)
        self.spec = spec

    def _resolve_cmap(self):
        """The plain named `spec.cmap` for temperature/stormwatch, or the live
        threshold-built colormap for ozone/pwat (spec.threshold_setting set) --
        reads the current threshold + palette settings each render, so config edits
        take effect without a restart."""
        spec = self.spec
        if not spec.threshold_setting:
            return __import__("matplotlib.cm", fromlist=["get_cmap"]).get_cmap(spec.cmap)

        threshold = float(
            self.settings.get(spec.threshold_setting, spec.threshold_default)
        )
        palette_name = self.settings.get("palette", spec.palette_default)
        palette_colors = spec.palettes.get(palette_name, spec.palettes[spec.palette_default])
        return _threshold_colormap(
            spec.vmin, spec.vmax, threshold, spec.focus, palette_colors, spec.flat_color
        )

    def plot(self, field0, state: ForecastState):
        """Render the static PNG (this hour) + global data texture for one scalar field.

        Consumes the per-hour field passed by render_all_hours (which fetches the
        correct hour and skips fresh ones), matching the precipitation pattern.
        """
        if not field0 or field0.get("values") is None:
            logger.warning(
                f"Skipping {self.section}: current-hour field not available in DB yet."
            )
            return

        logger.debug(
            f"Plotting {self.status_product} for {self.map_data.region.region_identifier}"
        )

        lats = field0["lat"]
        lons = field0["lon"]
        values = field0["values"]

        # LOD interpolation
        new_lats, new_lons, values_smooth = self.regrid_for_lod(values, lats, lons)

        output_path_for_hour = self.get_output_path_for_hour(state.fhour)

        # The static PNG (contourf) and the GPU data texture (encode_frames, below)
        # are independent outputs -- a contourf failure must not block the texture,
        # since that's what the frontend's animated WebGL layer actually reads.
        # contourf occasionally hits a known Cartopy bug (antimeridian-wrapping
        # polygon reprojection inside its own get_datalim() call, deep in
        # cartopy.crs._rings_to_multi_polygon -> shapely's MultiPolygon() rejecting a
        # nested multi-polygon) for specific, deterministic field topologies -- not
        # something we can prevent from our side without changing the render method
        # entirely. should_plot_for_hour will keep retrying next cycle, but a
        # persistently-triggering field (confirmed possible: the same hour's data can
        # trip this on every attempt) would otherwise never get its texture either.
        try:
            plot = Plot(self.map_data.region)
            plot.get_figure()

            cmap = self._resolve_cmap()
            norm = mcolors.Normalize(vmin=self.spec.vmin, vmax=self.spec.vmax)

            # Closes the antimeridian seam for this static render only -- see
            # close_lon_seam_for_contour's docstring. regrid_for_lod's own new_lons/
            # values_smooth stay untouched for any other use (there is none here: the
            # GPU data texture below is encoded from field0["values"], the native
            # grid, not this LOD-regridded one).
            contour_lons, contour_values = self.close_lon_seam_for_contour(new_lons, values_smooth)
            plot.ax.contourf(
                contour_lons,
                clamp_lats_to_mercator_limit(new_lats),
                contour_values,
                levels=20,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
                extend=self.spec.extend,
                zorder=2,
            )
            plot.save_figure(output_path_for_hour)

            plt_close = getattr(plot, "close", None)
            if callable(plt_close):
                plt_close()
        except Exception as e:
            logger.warning(
                f"{self.section}: static render f{state.fhour:03d} failed: {e}"
            )

        # --- WebGL single-hour data texture (one frame per forecast hour;
        # the frontend scrubber assembles the animation from consecutive hours) ---
        base, _ = os.path.splitext(output_path_for_hour)
        encode_frames(
            [field0["values"]], f"{base}_data.png", self.spec.vmin, self.spec.vmax
        )
        logger.info(f"Finished {self.section} texture f{state.fhour:03d}.")
