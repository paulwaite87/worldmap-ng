#!/usr/bin/env python3
import os
import logging
import warnings
import numpy as np
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.interpolate import RegularGridInterpolator

# Internal imports
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.texture import encode_frames
from .common import MapData, ForecastState
from .plotting import Plot, clamp_lats_to_mercator_limit
from .single_hour_scalar import SingleHourScalarUpdater

# Silence warnings
warnings.filterwarnings("ignore", message=".*missingValue.*")
logging.getLogger("cfgrib").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class PrecipitationUpdater(SingleHourScalarUpdater):
    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "precipitation", map_data)

        # Top of the precip scale (mm/hr). Must match the frontend shader's VMAX.
        # The data texture is sqrt-encoded against this, so most of the 8-bit range
        # is spent on the low rates where precip actually lives (see encode_frames).
        self.VMAX_PRECIP = 100.0
        # Below this, raw GFS PRATE is blur/quantization noise, not real precipitation
        # -- the model rarely outputs an exact 0.0, so residual convective-scheme
        # values (as small as ~1e-8 mm/hr) show up across most of the globe. Matches
        # the palette's own lowest defined band (LEVELS[0] below). Fixed, independent
        # of the user-configurable min_mm_hr slider: the frontend applies min_mm_hr
        # live on top of this floor, so u_min=0 means "any MEANINGFUL precipitation",
        # not "any nonzero noise".
        self.MEANINGFUL_PRECIP_MM_HR = 0.1
        # Final softening pass in _apply_meaningful_floor, in grid CELLS at whatever
        # resolution the texture is encoded at (independent of level_of_detail's own
        # upsample factor -- this is about smoothing the encoded grid's own texel
        # step for the GPU's benefit, not about the underlying meteorological
        # smoothing _smooth_global_field already does). Fixed rather than LOD-scaled:
        # at higher LOD the grid is already finer (less step to begin with), so the
        # same absolute cell-count softens proportionally less there, which is the
        # right direction anyway.
        #
        # 0.75 (this constant's original value) was tuned by eye at a moderate zoom
        # and looked fine there, but a native 0.25-deg grid cell is still tens of
        # screen pixels wide once a user zooms in close -- no bilinear-sampled amount
        # of a 0.75-cell blur meaningfully softens a transition at that scale, and the
        # native grid's steps were clearly visible again (reported live, at zoom ~7).
        # 2.0 (support radius ~6 cells, ~1.5deg at LOD1) trades a wider, softer halo
        # around each core for one that stays visually smooth much closer in to that
        # kind of zoom -- confirmed against the real pipeline, both in a raw decoded-
        # texture crop (organic falloff, no steps even cell-by-cell) and live in the
        # browser at zoom ~7-7.6.
        self.EDGE_SMOOTH_SIGMA_CELLS = 2.0

        self.PALETTES = {
            "standard": [
                (0.0, 1.0, 1.0),
                (0.0, 0.5, 1.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.5, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
            ],
            "ocean_blue": [
                (0.8, 0.9, 1.0),
                (0.6, 0.8, 1.0),
                (0.4, 0.6, 1.0),
                (0.2, 0.4, 1.0),
                (0.0, 0.2, 0.8),
                (0.0, 0.0, 0.6),
                (0.0, 0.0, 0.4),
            ],
            "high_contrast": [
                (0.0, 0.9, 0.0),
                (0.0, 0.6, 0.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.6, 0.0),
                (1.0, 0.0, 0.0),
                (0.7, 0.0, 0.0),
                (1.0, 0.0, 1.0),
            ],
        }

    def _render_settings_signature(self) -> str:
        """Render-relevant settings for the static per-hour PNG -- min_mm_hr/opacity/
        palette are all baked directly into it below (min_rate zeroes sub-threshold
        cells, alpha/palette_name build the colormap). Mirrors
        IsobarUpdater._render_settings_signature: without this, a settings-only edit
        never touches the output file's mtime or the data's updated_at, so
        should_plot_for_hour would silently never re-render an already-cached hour for
        it. The GPU data texture ignores all three (fixed floor, no palette) and gets
        over-invalidated along with the PNG, but plot() always regenerates both in one
        call anyway."""
        return self._settings_signature(
            {
                "min_mm_hr": self.settings.get("min_mm_hr", 0.1),
                "opacity": self.settings.get("opacity", 50),
                "palette": self.settings.get("palette", "standard"),
            }
        )

    def plot(self, field0, state: ForecastState):
        """Static region render (frame 0) + colourbar key + global N-frame texture.

        Now consumes pre-processed fields from the DB instead of opening GRIBs.
        Outputs are cached per-hour: {basename}_f{fhour:03d}.png
        """
        logger.debug(
            f"Plotting precipitation for {self.map_data.region.region_identifier}"
        )

        min_rate = self.settings.get("min_mm_hr", 0.1)
        alpha = float(self.settings.get("opacity", 50) / 100)
        palette_name = self.settings.get("palette", "standard")

        # --- Static region render (frame 0) ---
        lats = field0["lat"]
        lons = field0["lon"]
        prate = field0["values"].copy()
        prate[prate < min_rate] = 0.0

        # LOD interpolation (fill_value=0: gaps read as "no rain"). Level-of-detail
        # also drives the post-interpolation smoothing strength below.
        new_lats, new_lons, prate_smooth = self.regrid_for_lod(
            prate, lats, lons, fill_value=0
        )
        filter_sigma = {"high": 1.2, "medium": 0.8}.get(self.lod_desc, 0.0)

        # Setup Plotting
        plot = Plot(self.map_data.region)
        plot.get_figure()

        levels = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
        base_colors = self.PALETTES.get(palette_name, self.PALETTES["standard"])
        rgba_colors = [(*c, alpha) for c in base_colors]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "smooth_precip", rgba_colors, N=256
        )
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Render Heatmap Contour
        prate_smooth = gaussian_filter(prate_smooth, sigma=filter_sigma)
        # Closes the antimeridian seam for this static render only -- see
        # close_lon_seam_for_contour's docstring. regrid_for_lod's own new_lons/
        # prate_smooth stay untouched for any other use (there is none here: the GPU
        # data texture below is encoded from field0["values"], the native grid, not
        # this LOD-regridded one).
        contour_lons, contour_prate = self.close_lon_seam_for_contour(new_lons, prate_smooth)
        plot.ax.contourf(
            contour_lons,
            clamp_lats_to_mercator_limit(new_lats),
            contour_prate,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            extend="max",
            antialiased=True,
            zorder=2,
        )

        # Per-hour output path: precipitation_f003.png (for f003 forecast hour)
        output_path_for_hour = self.get_output_path_for_hour(state.fhour)
        plot.save_figure(output_path_for_hour)

        plt_close = getattr(plot, "close", None)
        if callable(plt_close):
            plt_close()

        # --- WebGL single-hour data texture (one frame per forecast hour;
        # the frontend scrubber assembles the animation from consecutive hours) ---
        # Smooth the GLOBAL field before encoding so the banded LUT produces smooth
        # band boundaries instead of tracing the raw 0.25-deg grid (the old static
        # render smoothed its regional clip the same way; the texture never did).
        # Floor below MEANINGFUL_PRECIP_MM_HR AFTER smoothing (not before) -- flooring
        # first still leaves a wide halo, since the blur below re-smears values back
        # below the floor around every real rain patch. A fixed floor, independent of
        # the user-adjustable min_mm_hr slider, so the frontend's u_min=0 ("any
        # MEANINGFUL precipitation") isn't diluted by a blur-noise halo.
        base, _ = os.path.splitext(output_path_for_hour)
        smoothed = self._smooth_global_field(
            field0["lat"], field0["lon"], field0["values"]
        )
        smoothed = self._apply_meaningful_floor(smoothed)
        encode_frames(
            [smoothed], f"{base}_data.png", 0.0, self.VMAX_PRECIP, transform="sqrt"
        )
        logger.info(
            f"Finished Precipitation texture "
            f"f{state.fhour:03d} ({self.lod_desc} smoothing)."
        )

    def _smooth_global_field(self, lats, lons, values):
        """Upsample + Gaussian-blur the global precip field for a smooth texture.

        Tied to level_of_detail (reusing that setting's 3 levels), bounded for a
        GLOBAL grid so the texture stays a sane size:
            LOD 1 (low):    1x native 0.25 deg, light blur   (~4 MB/hr)
            LOD 2 (medium): 2x -> 0.125 deg,    medium blur   (~17 MB/hr)
            LOD 3 (high):   3x -> 0.083 deg,     stronger blur (~37 MB/hr)
        Blur sigma scales with the upsample factor so the PHYSICAL smoothing radius
        (~1.2 native cells) stays roughly constant across levels. The GPU also
        bilinear-filters the texture at render time, so even 1x looks smooth.
        """
        lod = int(getattr(self, "level_of_detail", 1) or 1)
        if lod >= 3:
            factor, base_sigma = 3, 1.2
        elif lod == 2:
            factor, base_sigma = 2, 1.2
        else:
            factor, base_sigma = 1, 1.2

        arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)

        if factor > 1:
            # Bilinear upsample onto a regular factor-x denser global grid.
            lat_inc = lats[::-1] if (len(lats) > 1 and lats[0] > lats[-1]) else lats
            src = arr[::-1, :] if (len(lats) > 1 and lats[0] > lats[-1]) else arr
            fn = RegularGridInterpolator(
                (lat_inc, lons), src, bounds_error=False, fill_value=0.0
            )
            new_lats = np.linspace(
                lat_inc[0], lat_inc[-1], (len(lat_inc) - 1) * factor + 1
            )
            new_lons = np.linspace(lons[0], lons[-1], (len(lons) - 1) * factor + 1)
            mlat, mlon = np.meshgrid(new_lats, new_lons, indexing="ij")
            arr = fn((mlat, mlon)).astype(np.float32)
            if len(lats) > 1 and lats[0] > lats[-1]:
                arr = arr[::-1, :]  # restore north-first row order for the texture

        sigma = base_sigma * factor
        if sigma > 0:
            arr = gaussian_filter(arr, sigma=sigma)
        # Side effect (matches regrid_for_lod's self.lod_desc pattern): the blur radius
        # in grid CELLS at whatever resolution this returned, for _apply_meaningful_floor
        # to size its dilation from -- a fixed radius would be wrong at a different LOD's
        # grid spacing (LOD2/3 upsample first, so the same physical smoothing radius
        # spans more cells there than at LOD1's native grid).
        self._smooth_sigma_cells = sigma
        return arr

    def _apply_meaningful_floor(self, arr):
        """Zero out everywhere that isn't near a real precipitation core, smoothly
        fading the immediate surroundings of a core rather than hard-clipping.

        A pure per-cell value clip/fade (`arr < floor -> 0`, or even a smoothstep
        ramp over the same range) can't tell apart two very different populations of
        sub-floor value that both fall in the same range: widespread blur/quantization
        NOISE far from any real rain (as high as ~0.05mm/hr, per live measurement --
        comparable in magnitude to genuine rain-edge falloff, so a value-only test
        can't separate them) vs. the genuine, spatially-local smooth falloff the
        Gaussian blur above gives the true edge of a real core. A hard clip kills both
        (but the real-edge kill is blocky, quantized to the grid); a plain smoothstep
        dims both without eliminating either (verified live: coverage only dropped
        from ~85% to ~71%, not the ~18% a real "meaningful precipitation only" reading
        should give).

        So: identify real cores (>= floor) directly, dilate that mask by the blur's own
        3-sigma support radius (the distance beyond which the Gaussian's contribution
        is negligible) to capture just their genuine falloff halo, and zero
        EVERYTHING outside that halo outright -- no fade, no matter how large the
        residual value, since anything out there is noise by construction. Inside the
        halo, keep the smoothstep fade so the true edge itself is smooth rather than a
        hard cliff.
        """
        floor = self.MEANINGFUL_PRECIP_MM_HR
        sigma_cells = getattr(self, "_smooth_sigma_cells", 0.0)
        radius = max(2, int(round(3 * sigma_cells)))

        core = arr >= floor
        halo = binary_dilation(core, iterations=radius) if core.any() else core

        t = np.clip(arr / floor, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep: C1-continuous, 0 at 0 and 1 at floor
        result = np.where(halo, arr * t, 0.0)

        # `halo`'s own edge is still a hard, grid-quantized cutoff (0 the instant a
        # cell falls outside the dilated mask) -- at typical zoom this renders as a
        # visibly stepped/staircase boundary, since the frontend now samples bilinear
        # rather than bicubic (bicubic's smoother reconstruction rang/overshot right at
        # this same edge, which is a worse artifact -- see loadLayer()'s bicubic:false).
        # A small Gaussian pass here is safe from that overshoot risk (a Gaussian
        # kernel's weights are all positive and sum to 1, so it's a strict weighted
        # average -- it cannot produce a value outside the local input range, unlike
        # cubic reconstruction), so it just softens the hard step into a gradual ramp
        # over about a cell, for bilinear sampling to interpolate cleanly.
        return gaussian_filter(result, sigma=self.EDGE_SMOOTH_SIGMA_CELLS)