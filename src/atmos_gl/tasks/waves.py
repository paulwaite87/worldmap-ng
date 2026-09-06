#!/usr/bin/env python3
import os
import logging
import warnings
import numpy as np

# Internal imports
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.texture import encode_uv
from atmos_gl.lib.coastline import LandMaskCache, nearest_fill_and_regrid_uv
from .common import Updater, MapData, MultiHourRenderMixin, ForecastState

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Magnitude scale for the animated swell particle field (metres of significant wave
# height). The GPU layer clips |velocity| to this; pick a little above the tallest
# swell you care to distinguish. Must match VMAX_WAVES on the frontend (waves.js).
VMAX_WAVES = 8.0

# Fixed regrid step for waves' coastline-crispness pass, same reasoning and same
# empirically-timed value as SST's/currents' (see tasks/sst.py, tasks/currents.py):
# GFS-Wave's native 0.25 deg grid is coarser than this, and the true coastline mask
# needs a fine enough grid to snap to. Not a user setting, for the same reason.
#
# Finer steps were tried and reverted: the coastline mask itself now scales fine
# (lib/coastline.py's coastline_land_mask rasterizes GSHHG 'h' directly onto the grid,
# ~19s/~3.8GB even for a full-globe 162M-point 0.02deg grid), but the surrounding
# regrid pass (scipy RegularGridInterpolator building a full global mesh, for both u
# and v) does not: memory scales with grid point count, and isolated measurement
# (WavesUpdater._masked_uv called directly, not through the render pool) showed
# 0.08deg=(2251,4500) costs ~1.6GB peak RSS, while 0.04deg=(4501,9000) -- only 4x the
# points -- already climbs past 5GB and gets OOM-killed on this deployment's host
# (dmesg: `Out of memory: Killed process ... (python) ... anon-rss:5249264kB`; 0.02deg
# hit the same wall live too). 0.08 is the ceiling this host supports for a
# full-global regrid; going finer needs a lower-memory regrid path (float32, or tiling
# the interpolation instead of building one 162M-point mesh), not just a smaller step.
_WAVES_REGRID_STEP_DEG = 0.08


class WavesUpdater(Updater, MultiHourRenderMixin):
    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "Waves", map_data)

        # Per-hour velocity texture for the animated swell bars AND (via the frontend's
        # in-shader valueDecode) the heat fill -- both now read the SAME texture. The
        # data_collector stores a GFS-Wave u/v field per forecast hour in the fieldstore;
        # render_all_hours (in run()) writes waves_f{NNN}_data.png for each. The
        # "_data.png" entry tells the per-hour publish/staleness machinery what we emit.
        self.per_hour_outputs = ["_data.png"]
        self.status_product = "waves"
        # The land mask depends only on the (fixed) regrid geometry, so compute it once
        # per run and reuse for every hour. Keyed by grid shape. Every LandMaskCache
        # consumer (currents, waves) now shares one GSHHG 'h' geometry -- see
        # docs/adr/0013 (supersedes docs/adr/0011's Natural Earth precision limitation,
        # found live via waves' animated bars visibly crossing land on complex
        # coastlines like Northland NZ/Tasmania during candidate #7).
        self._land_mask = LandMaskCache("Waves")

    def _masked_uv(self, field0):
        """Regrid + true-coastline-mask u/v once, shared by BOTH the per-hour swell
        texture (particles) and, via the frontend's in-shader valueDecode, the heat
        fill -- one pass now serves what used to be two separate masking mechanisms
        (native-NaN-only for particles, a live per-tile-pixel STRtree cut for the
        heat tiles). Same technique SST/currents use: nearest-fill native NaN first
        (GFS-Wave's own no-data-over-land) so bilinear interpolation doesn't bleed
        outward from the coast into legitimate open water, regrid to
        _WAVES_REGRID_STEP_DEG, then cut the true coastline.

        Returns (new_lats, u, v) -- new_lats is passed straight through to encode_uv
        for correct north-at-top row orientation (see encode_uv's docstring).
        """
        new_lats, new_lons, u, v = nearest_fill_and_regrid_uv(
            self.regrid_for_lod, field0["u"], field0["v"],
            field0.get("lat"), field0.get("lon"),
            step_deg=_WAVES_REGRID_STEP_DEG,
        )

        # LandMaskCache.get() (-> coastline_land_mask()) already dilates the mask by one
        # cell before returning it (see that function's own docstring) -- needed because
        # both consumers of this texture, the heat-fill shader's LINEAR-filtered discard
        # and the bar particle engine's VEL_SAMPLE, blend colour/velocity across the true
        # coastline edge otherwise.
        land = self._land_mask.get(new_lats, new_lons, u.shape)
        if land is not None and land.shape == u.shape:
            u[land] = np.nan
            v[land] = np.nan

        return new_lats, u, v

    def plot_swell(self, field0, state: ForecastState):
        """Write the per-hour swell velocity texture (R=U east, G=V north) from a
        fieldstore field. The collector already derived u/v from swh + wave direction
        (see waves_data_unpack); _masked_uv regrids + cuts the true coastline before
        encoding -- this is the animated-bars analogue of currents.plot, called once
        per catalog hour by render_all_hours. Land/no-data cells in u/v become
        transparent (alpha 0) so bars respawn there and the heat fill (which decodes
        speed from this same texture client-side) shows nothing. Separate from
        _write_velocity_texture (writes the single static base texture; kept for the
        forecast_stepping=off path)."""
        new_lats, u, v = self._masked_uv(field0)
        out_for_hour = self.get_output_path_for_hour(state.fhour)
        base, _ = os.path.splitext(out_for_hour)
        encode_uv(u, v, f"{base}_data.png", VMAX_WAVES, lat=new_lats)
        logger.info(
            f"Waves: wrote swell velocity texture f{state.fhour:03d}."
        )

    def _write_velocity_texture(self, field0):
        """Encode the now-hour swell vector field into <outfile_base>_data.png for the
        static (forecast_stepping=off) particle layer. u/v already come from the collector
        (direction = wave direction, magnitude = significant wave height, so taller swell
        drifts faster), already wrapped to a clean -180..180 equirect grid by the unpacker;
        _masked_uv regrids + cuts the true coastline the same way plot_swell does --
        the static-base analogue of its per-hour textures. Land/no-data cells stay
        transparent (alpha 0)."""
        new_lats, u, v = self._masked_uv(field0)
        base, _ = os.path.splitext(self.output_path)
        encode_uv(u, v, f"{base}_data.png", VMAX_WAVES, lat=new_lats)
        logger.info("Waves: wrote swell velocity texture for the animated layer.")

    def run(self, max_hours=None):
        # Warms the shared per-cycle GFS baseline cache (map_data.shared_state) for
        # other updaters this cycle; both sections below resolve their own state from
        # the catalog, so the return value here is unused.
        self.get_gfs_state()

        # 1) Per-hour swell velocity textures for the animated bars, from the fieldstore.
        # Done FIRST and unconditionally, mirroring how wind/currents render every
        # catalog hour; gap-fills only missing/stale hours. max_hours=1 from
        # layer_builder's round-robin dispatch renders one hour and returns, so this
        # layer doesn't monopolise a render-pool worker.
        plotted = self.render_all_hours(
            "waves",
            plot_fn=self.plot_swell,
            field_ready=lambda f: f.get("u") is not None and f.get("v") is not None,
            max_hours=max_hours,
        )

        # 2) Legend key + the static (forecast_stepping=off) base texture, from the
        # fieldstore now-hour field. The collector stores fhour_0..end, so the earliest
        # catalog hour IS the now-hour.
        resolved = self.latest_store_run(["waves"])
        if not resolved:
            logger.warning(
                "Waves: no waves field in the fieldstore yet (collector hasn't run); "
                "skipping static texture."
            )
            return plotted
        run_date, run_id, hours = resolved
        now_fh = hours[0]
        state = ForecastState.at_hour(run_date, run_id, now_fh)
        field0 = self.get_db_field_at_hour(state, "waves")
        if not field0 or field0.get("u") is None or field0.get("v") is None:
            logger.warning(
                "Waves: now-hour field missing u/v in the fieldstore; skipping static texture."
            )
            return plotted

        # No version-gate needed now (that existed to skip the expensive tile pyramid
        # warm-up) -- _write_velocity_texture is just one more encode_uv call, cheap
        # enough to run unconditionally every cycle.
        self._write_velocity_texture(field0)
        return plotted