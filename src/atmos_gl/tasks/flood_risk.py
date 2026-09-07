#!/usr/bin/env python3
"""Flood Risk layer rendering (Updater) -- see issue #371 and its follow-up
grilling (Live mode's data-source pivot, collectors/flood_risk.py's module
docstring).

Historical and Live are two INDEPENDENTLY-SOURCED metrics, not two views of the
same data: Live's `band` is MODIS's binary flood detection (0/1, see
lib/flood_risk.py's resample_modis_flood_tile_onto_grid); Historical's `band` is
JRC's depth-HAZARD category (0..4, JRC's own reclass scale). Each therefore gets
its own encode domain (_LIVE_ENCODE_DOMAIN / _HISTORICAL_ENCODE_DOMAIN) -- the two
are not comparable on one shared scale.

Both render as a raw, un-colored data texture (issue #312's client-side-palette
convention, matching greenhouse_gases/air_quality) -- there is no separate static
contourf PNG for either mode, since contourf's smooth interpolation between
levels is the wrong visual for discrete category data in the first place, and
this layer's OUTFILES entry (a plain ".png") is therefore itself the texture,
exactly like greenhouse_gases's own canonical output.

Both variants are now single-shot cached rasters (mirroring _render_historical's
own shape) -- Live used to be a per-forecast-hour animated series back when it
sourced a GloFAS forecast, but MODIS reports a single continuously-refreshed
"current" state with no forecast-hour dimension at all, so render_all_hours'
multi-hour scrubbing machinery no longer applies. Both render EVERY cycle,
independent of the configured mode -- mirrors GhgUpdater's "render everything,
publish only what's selected" pattern, so switching modes in the UI is instant
rather than waiting for a fresh render.
"""
import logging
import os

import numpy as np

from atmos_gl.lib.coastline import LandMaskCache
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.flood_risk import (
    jrc_hazard_mosaic_cache_path,
    load_jrc_hazard_mosaic,
    modis_flood_mosaic_cache_path,
)
from atmos_gl.lib.texture import encode_frames
from .common import MapData, Updater

logger = logging.getLogger(__name__)

# Live's band is binary (0 = no flood, 1 = MODIS Flood pixel value 3) -- see
# lib/flood_risk.py's resample_modis_flood_tile_onto_grid.
_LIVE_ENCODE_DOMAIN = (0.0, 1.0)
# Historical's band is JRC's own reclass scale (1:<1m .. 4:>10m) plus this
# mosaic's own 0 ("no known hazard" / outside any tile's footprint) -- see
# lib/flood_risk.py's resample_jrc_tile_onto_grid docstring.
_HISTORICAL_ENCODE_DOMAIN = (0.0, 4.0)


class FloodRiskUpdater(Updater):
    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "flood_risk", map_data)
        self.mode = self.settings.get("mode", "live").strip().lower()
        # Both modes are single-shot cached rasters now (no forecast-hour series
        # for either), so status_product stays unset -- layer_status() falls back
        # to the decaying-freshness formula, same as sst/clouds/markers.
        self._land_mask_cache = LandMaskCache("FloodRiskLive")

    def _variant_path(self, suffix: str) -> str:
        base, ext = os.path.splitext(self.output_path)
        return f"{base}_{suffix}{ext}"

    def _render_live(self) -> str | None:
        """Render the MODIS-observed flood mosaic (rebuilt by FloodRiskLiveCollector
        every cycle it finds a changed or newly-expired tile) into its own "_live"
        variant path, if it's cached and the render isn't already fresh. Mirrors
        _render_historical exactly -- both modes are single continuously-refreshed
        cached rasters, not per-forecast-hour series.

        Cut to land via LandMaskCache before encoding: resample_modis_flood_tile_
        onto_grid's own "flood pixel near surface water" rule doesn't distinguish
        sea from river/lake water, so a coastal sea inlet (real-world example: the
        Marlborough Sounds, NZ) reads as "near water" just by being sea, and gets
        flagged as flooded the same as an actual flooded river. The true coastline
        cut removes that class of false positive the same way currents/waves/
        FireWeatherUpdater already use it for theirs.

        dilate=False (unlike every other LandMaskCache caller): the default
        dilate=True GROWS land by one cell so a bilinear-filtered GPU texture's
        coastline blend zone never bleeds colour onto land (see
        coastline_land_mask's own docstring) -- the opposite of what this layer
        needs. Growing land shrinks the water mask, which at this mosaic's ~0.05deg
        (~5km) resolution reclassified real narrow-channel sea cells (confirmed
        live: the Marlborough Sounds) as land, letting their false-positive flood
        value through unmasked. dilate=False keeps the raw geometric
        classification -- still limited by the mosaic's own grid resolution for
        the very narrowest channels, but no longer making that worse.

        exclude_lakes=True: GSHHG's L1 coastline tier has no concept of an inland
        water body, so a lake reads as plain "land" and its MODIS-flagged surface
        (same "near surface water" false-positive as the sea/river case above)
        survives an ocean-only mask unmasked -- confirmed live over real lakes.
        exclude_lakes=True additionally cuts GSHHG's L2 lake polygons (see
        coastline_land_mask's own docstring)."""
        mosaic_path = modis_flood_mosaic_cache_path(self.workdir)
        if not os.path.exists(mosaic_path):
            return None

        out = self._variant_path("live")
        sig = self._settings_signature({})
        if not self._is_render_fresh(out, [mosaic_path], sig):
            band, lat, lon = load_jrc_hazard_mosaic(mosaic_path)
            land = self._land_mask_cache.get(
                lat, lon, band.shape, dilate=False, exclude_lakes=True
            )
            if land is not None:
                band = band.copy()
                band[~land] = 0
            encode_frames([band.astype(np.float32)], out, *_LIVE_ENCODE_DOMAIN)
            self._write_render_signature(out, sig)
            logger.info(f"{self.section}: rendered live inundation texture.")
        return out

    def _render_historical(self) -> str | None:
        """Render the static JRC hazard mosaic's data texture into its own "_historical"
        variant path, if the mosaic is cached and the render isn't already fresh (the
        mosaic itself never changes once FloodRiskHistoricalCollector finishes it, so
        this effectively renders once, ever). Returns the variant path, or None if the
        mosaic isn't cached yet (the historical collector hasn't finished downloading
        all 271 tiles)."""
        mosaic_path = jrc_hazard_mosaic_cache_path(self.workdir)
        if not os.path.exists(mosaic_path):
            return None

        out = self._variant_path("historical")
        sig = self._settings_signature({})
        if not self._is_render_fresh(out, [mosaic_path], sig):
            band, _lat, _lon = load_jrc_hazard_mosaic(mosaic_path)
            encode_frames([band.astype(np.float32)], out, *_HISTORICAL_ENCODE_DOMAIN)
            self._write_render_signature(out, sig)
            logger.info(f"{self.section}: rendered historical hazard texture.")
        return out

    def run(self, max_hours=None):
        # max_hours is a no-op here -- both modes render once per cycle, not per
        # forecast hour, so it has nothing to cap. Accepted only so layer_builder's
        # dispatch can call every TASK_CLASSES entry's run() the same way (see
        # MarkerUpdater.run's identical convention/comment).
        live_out = self._render_live()
        historical_out = self._render_historical()

        # Publish whichever mode is currently configured to the stable,
        # run-agnostic base filename the frontend reads.
        if self.mode == "historical":
            if historical_out:
                self._publish_variant(historical_out)
        elif live_out:
            self._publish_variant(live_out)
