#!/usr/bin/env python3
import os
import logging

import numpy as np

from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.texture import encode_uv
from .common import MapData, ForecastState
from .vector_field import VectorFieldUpdater
from atmos_gl.lib.coastline import LandMaskCache, nearest_fill_and_regrid_uv

logger = logging.getLogger(__name__)

# Fixed regrid step for currents' coastline-crispness pass, same reasoning and same
# empirically-timed value as SST's _SST_REGRID_STEP_DEG (see tasks/sst.py): RTOFS's
# native ~0.1 deg server-side regrid is itself coarser than this, so the true
# coastline mask -- cut on the NATIVE grid -- was snapping to ~11km blocks. Not a
# user setting, for the same reason SST's isn't.
_CURRENTS_REGRID_STEP_DEG = 0.08


class CurrentsUpdater(VectorFieldUpdater):
    """Ocean surface currents from RTOFS (via the fieldstore).

    Rebuilt for the GPU pipeline: the data_collector downloads RTOFS, regrids the
    tripolar u/v to a regular lat/lon grid, and stores it in the fieldstore. This
    task just reads that per-hour u/v field and writes ONE velocity texture per hour
    (R=U east, G=V north) — the same encoding wind uses. The frontend then renders
    BOTH a speed fill and advected particles from that single texture (speed is
    sqrt(u^2+v^2) computed in-shader, so no separate speed texture is needed).

    Reads use the RTOFS run (get_rtofs_state), NOT the GFS run.
    """

    VMAX = 2.5  # m/s; clips the strongest currents (Gulf Stream/Kuroshio ~2.5)

    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "Currents", map_data)
        # The land mask depends only on the (fixed) regrid geometry, so compute it once
        # per run and reuse for every hour. Keyed by grid shape.
        self._land_mask = LandMaskCache("Currents")

    def _warm_baseline_cache(self):
        self.get_rtofs_state()

    def plot(self, field0, state: ForecastState):
        """Write the per-hour current velocity texture (R=U east, G=V north).

        First regrid u/v up from RTOFS's native ~0.1 deg grid to
        _CURRENTS_REGRID_STEP_DEG, so the coastline cut below has a fine enough grid
        to snap to (cutting on the coarser native grid left ~11km blocky coastlines).
        Then (1) drop water slower than current_speed_minimum (m/s) and (2) cut land
        out with true coastline geometry. Both become NaN -> alpha 0 in the texture,
        which hides them in BOTH layers at once: the fill shader discards alpha<0.5
        texels, and the particle engine treats alpha<0.5 as land (no spawn / reset).
        So one data-side mask removes slow water and land encroachment from the speed
        fill and the flowing particles together.
        """
        # RTOFS's native NaN (land cells, ~33% of the grid) is nearest-filled before
        # regridding -- same technique SST uses -- so bilinear interpolation doesn't
        # bleed NaN outward from the coast into legitimate near-shore water; the true
        # coastline mask below is what actually determines land/sea, not this fill.
        new_lats, new_lons, u, v = nearest_fill_and_regrid_uv(
            self.regrid_for_lod, field0["u"], field0["v"],
            field0.get("lat"), field0.get("lon"),
            step_deg=_CURRENTS_REGRID_STEP_DEG,
        )

        # (1) Speed-minimum threshold (m/s). Below this -> no display. 0 disables.
        try:
            speed_min = float(self.settings.get("current_speed_minimum", 0.0))
        except (TypeError, ValueError):
            speed_min = 0.0
        if speed_min > 0.0:
            with np.errstate(invalid="ignore"):
                below = np.hypot(u, v) < speed_min   # NaN compares False -> left as-is
            u[below] = np.nan
            v[below] = np.nan

        # (2) Coastline cut: remove ocean values that the regrid smeared onto land.
        # LandMaskCache.get() (-> coastline_land_mask()) already dilates the mask by one
        # cell before returning it (see that function's own docstring) -- needed because
        # both consumers of this texture, the fill shader's LINEAR-filtered discard and
        # the particle engine's VEL_SAMPLE, blend colour/velocity across the true
        # coastline edge otherwise.
        land = self._land_mask.get(new_lats, new_lons, u.shape)
        if land is not None and land.shape == u.shape:
            u[land] = np.nan
            v[land] = np.nan

        out_for_hour = self.get_output_path_for_hour(state.fhour)
        base, _ = os.path.splitext(out_for_hour)
        encode_uv(u, v, f"{base}_data.png", self.VMAX, lat=new_lats)

        logger.info(
            f"Finished Currents velocity texture "
            f"f{state.fhour:03d} (R=U, G=V)."
        )