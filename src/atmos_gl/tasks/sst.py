#!/usr/bin/env python3
import os
import logging
import gc
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

# Internal imports
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.coastline import coastline_land_mask
from atmos_gl.lib.texture import encode_frames
from .common import Updater, MapData

logger = logging.getLogger(__name__)

# Fixed regrid step for SST's coastline-crispness pass -- finer than any regrid_for_lod
# LOD tier gives (needed since OISST's native 0.25 deg/~28km grid is much coarser than
# other fields), not a user setting (SST's raw data doesn't warrant detail control).
# Chosen empirically: at world scale 0.05 deg (~5.6km) took ~4 minutes for regrid+mask
# alone (25.9M points) -- impractical for a periodic render. 0.08 deg (~8.9km) completes
# a full render (regrid+mask+encode) in well under that, comparable to the old 0.15 deg
# tier's production timing, while resolving roughly twice as fine.
_SST_REGRID_STEP_DEG = 0.08

# Fixed physical domains encode_frames normalises into, for the raw client-LUT texture
# (issue #312) -- deliberately NOT the user's live min_c/max_c (those are applied
# entirely client-side now, see ui/modules/sst.js's buildScaledLUT, so a palette/scale
# change never needs a server re-render). Absolute mode's domain matches the min_c/
# max_c slider's own hard bounds (routes/field_specs.py's _MIN_MAX_C: 0-36 DegC)
# exactly, so no realistic user-chosen display range can fall outside it. Anomaly's
# domain has no such natural bound (it's auto-scaled from live data, floored at 0.5 --
# see plot() below); -10..10 DegC is a generous margin over that in practice, so only
# a genuinely extreme anomaly would clip.
_ABS_ENCODE_VMIN, _ABS_ENCODE_VMAX = 0.0, 36.0
_ANOMALY_ENCODE_VMIN, _ANOMALY_ENCODE_VMAX = -10.0, 10.0


class SSTUpdater(Updater):
    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "sst", map_data)
        self.mode = self.settings.get("mode", "absolute").strip().lower()

    def _output_path_for_mode(self, mode: str) -> str:
        """Per-mode, ALWAYS-kept-fresh output path: 'data/sst.png' -> e.g.
        'data/sst_anomaly.png'. Both modes render here every cycle (independent of
        the configured `mode`) so the frontend can switch between them instantly --
        see ui/modules/sst.js. Since #312, this path holds a raw, un-colored data
        texture (encode_frames), not a colored image -- the palette/LUT is applied
        entirely client-side, reading the same path."""
        base, ext = os.path.splitext(self.output_path)
        return f"{base}_{mode}{ext}"

    def plot(self, mode: str, nc_path: str, output_path: str):
        # --- Data Loading ---
        ds = xr.open_dataset(nc_path, chunks={"time": 1})
        latest_slice = ds.isel(time=-1)

        lat_raw = latest_slice["lat"].values
        lon_raw = latest_slice["lon"].values

        # Dynamically target 'anom' for anomaly mode, or 'sst' for absolute mean mode
        data_var = "anom" if mode == "anomaly" else "sst"
        raw_matrix = latest_slice[data_var].values.squeeze()

        # Cleanly transform NOAA's 0-360 range into a standard -180 to +180 baseline
        lon_norm = ((lon_raw + 180) % 360) - 180

        # Sort along longitudes to avoid geometric rendering seams or distortions
        lon_sort_idx = np.argsort(lon_norm)
        lon_norm = lon_norm[lon_sort_idx]
        raw_matrix = raw_matrix[:, lon_sort_idx]

        ds.close()
        del ds
        gc.collect()

        # Nearest-fill NaN (native OISST land cells) before regridding -- same technique
        # raster_tiles.bake_field uses for waves/wind tiles -- so the interpolation below
        # doesn't blur a blank halo around the coast; the true coastline geometry below
        # is what actually determines the rendered land/sea boundary, not this fill.
        raw_matrix = np.asarray(raw_matrix, dtype=np.float64)
        bad = ~np.isfinite(raw_matrix)
        if bad.any() and not bad.all():
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            raw_matrix = raw_matrix[tuple(idx)]

        # LOD regrid off OISST's coarse native 0.25 deg grid down to _SST_REGRID_STEP_DEG,
        # then cut the true coastline the same way currents.py does -- see
        # docs/adr/0004-render-bbox-clipping-is-dead-code.md.
        new_lats, new_lons, display_data = self.regrid_for_lod(
            raw_matrix, lat_raw, lon_norm, fill_value=np.nan,
            step_override=_SST_REGRID_STEP_DEG,
        )
        # regrid_for_lod always returns ASCENDING (south-first) latitude rows,
        # regardless of the input's own order (see its docstring) -- but the GPU fill
        # shader (ui/modules/_webglfill.js's VS_BODY: "y in [0,1] lat north->south")
        # requires row 0 = north pole, the same contract every other encode_frames
        # texture relies on (see precipitation.py's _smooth_global_field, which
        # explicitly flips back with the comment "restore north-first row order for
        # the texture"). This restore was missing here, so the whole SST layer --
        # data AND land mask alike -- rendered mirrored across the equator, reported
        # live as "the landmass mask is upside-down".
        new_lats = new_lats[::-1]
        display_data = display_data[::-1, :]
        mesh_lon, mesh_lat = np.meshgrid(new_lons, new_lats)
        # coastline_land_mask() dilates the mask by one cell before returning (see its
        # own docstring) -- needed because the GPU fill layer samples this texture with
        # LINEAR filtering and a hard alpha>=0.5 discard, which would otherwise blend
        # colour across the true coastline edge onto land.
        land = coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)
        if land is not None and land.shape == display_data.shape:
            display_data[land] = np.nan

        if mode == "anomaly":
            # Isolates 98th percentile of absolute variance for stable scale bounds --
            # the LIVE display range a signed-in user's anomaly legend/LUT remaps onto,
            # written to sst_meta.json below (not the fixed encode domain above).
            abs_anomalies = np.abs(display_data)
            calculated_range = (
                float(np.nanpercentile(abs_anomalies, 98))
                if np.any(~np.isnan(abs_anomalies))
                else 4.0
            )
            anomaly_range = max(0.5, calculated_range)
            vmin, vmax = -anomaly_range, anomaly_range
            encode_vmin, encode_vmax = _ANOMALY_ENCODE_VMIN, _ANOMALY_ENCODE_VMAX
        else:
            vmin = self.settings.get("min_c", 0)
            vmax = self.settings.get("max_c", 32)
            encode_vmin, encode_vmax = _ABS_ENCODE_VMIN, _ABS_ENCODE_VMAX

        # Raw, un-colored data texture (issue #312) -- land cells stay NaN (encoded as
        # alpha=0, discarded by the fragment shader), same convention currents.py/
        # waves.py's land masking already relies on. The palette/LUT and the live
        # min_c/max_c (or anomaly vmin/vmax) display range are applied entirely
        # client-side (ui/modules/sst.js), reading this fixed, generous physical
        # domain -- so a palette or scale change never needs a server re-render.
        encode_frames([display_data], output_path, encode_vmin, encode_vmax)

        # Legend key renders entirely client-side too (issue #302). Absolute mode's
        # vmin/vmax come straight from settings (min_c/max_c, already visible to the
        # frontend via /api/config); anomaly mode's are auto-scaled from live data
        # (98th percentile, above) and have no other way to reach the frontend, so
        # only anomaly is written -- mirrors GhgUpdater's identical sst_meta.json ->
        # ghg_meta.json sidecar for the same "data-dependent range" problem.
        if mode == "anomaly":
            self._write_meta_sidecar("sst_meta.json", mode, {"vmin": vmin, "vmax": vmax})

        logger.debug(f"Successfully rendered raw NOAA OISST texture in {mode} mode.")

    def _mode_settings_signature(self, mode: str) -> str:
        """Render-relevant settings for `mode`, for _is_render_fresh. Since #312,
        opacity/palette/min_c/max_c apply entirely client-side (the encoded texture's
        domain is a fixed constant -- see plot()), so none of them change the encoded
        pixels anymore; this is deliberately empty and freshness is governed by
        source-data mtime alone (_is_render_fresh's other check). Kept as a real call
        (not inlined `""`) so a future setting that DOES need to force a re-render has
        somewhere to add itself."""
        return self._settings_signature({})

    def run(self, max_hours=None):
        # max_hours is a no-op here -- SST renders once per cycle, not per forecast
        # hour, so it has nothing to cap. Accepted only so layer_builder's dispatch can
        # call every TASK_CLASSES entry's run() the same way.
        # The data_collector fetches BOTH modes' netCDFs unconditionally (see
        # collectors/sst.py), so both are rendered here every cycle too -- each to its
        # own permanent output path, independent of which mode is currently configured
        # -- so the frontend can switch between them instantly (ui/modules/sst.js)
        # rather than waiting on a config change + re-render round-trip.
        from atmos_gl.lib.oisst import oisst_cache_path, OISST_MODES

        for mode in OISST_MODES:
            nc_path = oisst_cache_path(self.workdir, mode)
            if not os.path.exists(nc_path):
                logger.info(
                    f"SST: cache {os.path.basename(nc_path)} not present yet "
                    f"(data collector hasn't fetched it); skipping {mode}."
                )
                continue

            out = self._output_path_for_mode(mode)
            sig = self._mode_settings_signature(mode)
            fresh = self._is_render_fresh(out, [nc_path], sig)
            if not fresh:
                logger.info(f"Generating SST {mode} texture...")
                self.plot(mode, nc_path, out)
                self._write_render_signature(out, sig)

            if mode == self.mode:
                self._publish_variant(out)
