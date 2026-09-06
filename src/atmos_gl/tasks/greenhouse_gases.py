#!/usr/bin/env python3
import gc
import logging
import os

import numpy as np

from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.lib.coastline import coastline_land_mask
from atmos_gl.lib.greenhouse_gases import (
    SPECIES,
    camsforecast_cache_path,
    compute_anomaly,
    egg4_baseline_cache_path,
    resolve_baseline_year,
)
from atmos_gl.lib.netcdf_field import load_field
from atmos_gl.lib.texture import encode_frames
from .common import Updater, MapData

logger = logging.getLogger(__name__)

# CAMS's high-resolution forecast is ~9km (~0.1 deg) native, but rendering AT that
# resolution is far too slow to be practical: live timing found the old pcolormesh
# render alone taking >80s per render at the native 6.5M-point grid (regrid+coastline-
# mask together are a comparatively cheap ~7s) -- with 4 species x mode combinations
# rendered every cycle, that's minutes per cycle just for this one layer. 0.25 deg
# (matching this codebase's "low" LOD tier default) cuts the point count by ~6x,
# bringing regrid+encode back into the same ballpark as every other layer.
_REGRID_STEP_DEG = 0.25

# Both CAMS datasets (the current forecast and the EGG4 baseline) use the same
# in-file netCDF variable names for these two species -- confirmed by downloading and
# inspecting real files from both datasets (see the published spec's issue comments).
# NOT the same as the request-time CDS variable identifiers
# (co2_column_mean_molar_fraction/ch4_column_mean_molar_fraction, used in
# collectors/greenhouse_gases.py's request builders) -- ECMWF's GRIB-to-netCDF
# conversion exposes them under their short GRIB_cfVarName instead. Units are
# confirmed directly from the files' own `units` attribute: ppm/ppb, exactly the
# display units this layer wants, so no conversion factor is needed.
_CAMS_VARS = {"co2": "tcco2", "ch4": "tcch4"}

# Fixed physical domains encode_frames normalises into, for the raw client-LUT texture
# (issue #312) -- deliberately NOT the user's live co2/ch4 min/max settings (those are
# applied entirely client-side now, see ui/modules/greenhouse_gases.js's
# buildScaledLUT, so a palette/scale change never needs a server re-render). Absolute
# mode's domains match each species' own min/max slider's hard bounds
# (routes/field_specs.py: co2 380-450ppm, ch4 1600-2100ppb) exactly, so no realistic
# user-chosen display range can fall outside them. Anomaly's domains have no such
# natural bound (auto-scaled from live data, floored below) -- both gases are well
# mixed, so an old baseline_year (as far back as 2003, ~23 years of rise) shows up as a
# near-globally-uniform offset rather than a small localised anomaly the way SST's
# weather-driven anomaly does; these margins are generous over that worst case.
_ABS_ENCODE_DOMAIN = {"co2": (380.0, 450.0), "ch4": (1600.0, 2100.0)}
_ANOMALY_ENCODE_DOMAIN = {"co2": (-100.0, 100.0), "ch4": (-300.0, 300.0)}

# Flat, species-prefixed setting keys (co2_min_ppm, ch4_palette, ...) rather than a
# nested co2/ch4 sub-dict -- FIELD_SPECS/validate_against_specs (routes/field_specs.py)
# only understands flat (section, option) keys, matching every other section's
# settings shape (e.g. sst.min_c, waves.min_wave_height), so per-species settings stay
# flat with a species prefix instead of introducing nested-dict config support.
_SCALE_SETTING_KEYS = {"co2": ("co2_min_ppm", "co2_max_ppm"), "ch4": ("ch4_min_ppb", "ch4_max_ppb")}


class GhgUpdater(Updater):
    def __init__(self, config: AtmosGLConfig, map_data: MapData):
        super().__init__(config, "greenhouse_gases", map_data)
        self.species = self.settings.get("species", "co2").strip().lower()
        self.mode = self.settings.get("mode", "absolute").strip().lower()

    def _output_path_for(self, species: str, mode: str) -> str:
        """Per-(species, mode), ALWAYS-kept-fresh output path: 'data/greenhouse_gases.png'
        -> e.g. 'data/greenhouse_gases_co2_anomaly.png'. All 4 combinations render here
        every cycle (independent of the configured species/mode) so the frontend can
        switch between them instantly -- see ui/modules/greenhouse_gases.js. Since #312,
        this path holds a raw, un-colored data texture (encode_frames), not a colored
        image -- the palette/LUT is applied entirely client-side, reading this path."""
        base, ext = os.path.splitext(self.output_path)
        return f"{base}_{species}_{mode}{ext}"

    def plot(self, species: str, mode: str, current_nc: str, egg4_nc: str | None, output_path: str):
        display_data, lat_raw, lon_norm = load_field(current_nc, _CAMS_VARS[species])

        if mode == "anomaly":
            baseline_matrix, baseline_lat, baseline_lon = load_field(
                egg4_nc, _CAMS_VARS[species], reduce="mean"
            )
            display_data = compute_anomaly(
                display_data, lat_raw, lon_norm, baseline_matrix, baseline_lat, baseline_lon
            )

        new_lats, new_lons, display_data = self.regrid_for_lod(
            display_data, lat_raw, lon_norm, fill_value=np.nan, step_override=_REGRID_STEP_DEG,
        )
        mesh_lon, mesh_lat = np.meshgrid(new_lons, new_lats)
        # coastline_land_mask() dilates the mask by one cell before returning (see its
        # own docstring) -- needed because this renders through the same GPU fill
        # layer's LINEAR-filtered alpha discard SST does, and would otherwise show the
        # same coastal colour bleed SST had before docs/adr/0014.
        land = coastline_land_mask(mesh_lon, mesh_lat, -180.0, -90.0, 180.0, 90.0)
        if land is not None and land.shape == display_data.shape:
            display_data[land] = np.nan

        if mode == "anomaly":
            # Auto-scaled from the data (98th percentile of |anomaly|) rather than a
            # manual setting -- anomaly ranges are small and data-dependent enough
            # that a fixed scale would need constant retuning. Same technique
            # SSTUpdater.plot() uses for SST's anomaly mode. This is the LIVE display
            # range a signed-in user's anomaly legend/LUT remaps onto, written to
            # ghg_meta.json below (not the fixed encode domain above).
            abs_anomalies = np.abs(display_data)
            calculated_range = (
                float(np.nanpercentile(abs_anomalies, 98))
                if np.any(~np.isnan(abs_anomalies))
                else 1.0
            )
            anomaly_range = max(0.1, calculated_range)
            vmin, vmax = -anomaly_range, anomaly_range
            encode_vmin, encode_vmax = _ANOMALY_ENCODE_DOMAIN[species]
        else:
            min_key, max_key = _SCALE_SETTING_KEYS[species]
            vmin = self.settings.get(min_key, 0)
            vmax = self.settings.get(max_key, 1)
            encode_vmin, encode_vmax = _ABS_ENCODE_DOMAIN[species]

        # Raw, un-colored data texture (issue #312) -- land cells stay NaN (encoded as
        # alpha=0, discarded by the fragment shader), same convention currents.py/
        # waves.py's land masking already relies on. The palette/LUT and the live
        # min/max (or anomaly vmin/vmax) display range are applied entirely
        # client-side (ui/modules/greenhouse_gases.js), reading this fixed, generous
        # physical domain -- so a palette or scale change never needs a server
        # re-render.
        encode_frames([display_data], output_path, encode_vmin, encode_vmax)

        # Legend key renders entirely client-side too (issue #302). Absolute mode's
        # vmin/vmax come straight from settings (co2_min_ppm/co2_max_ppm etc, already
        # visible to the frontend via /api/config); anomaly mode's are auto-scaled from
        # live data (98th percentile, above) and have no other way to reach the
        # frontend, so only anomaly is written -- mirrors SSTUpdater's identical
        # sst_meta.json sidecar for the same "data-dependent range" problem.
        if mode == "anomaly":
            self._write_meta_sidecar(
                "ghg_meta.json", species, {"anomaly": {"vmin": vmin, "vmax": vmax}}
            )

        gc.collect()

        logger.debug(f"Successfully rendered {species} {mode} greenhouse gas texture.")

    def _mode_settings_signature(self, species: str, mode: str) -> str:
        """Render-relevant settings for (species, mode), for _is_render_fresh. Since
        #312, opacity/palette/min/max apply entirely client-side (the encoded
        texture's domain is a fixed constant per species -- see plot()), so none of
        them change the encoded pixels anymore for either mode. baseline_year is kept
        for anomaly: it's not a colour/scale setting -- it selects which EGG4 source
        file gets diffed against, so a baseline_year change genuinely changes the
        computed VALUES even though the source netCDF paths' mtimes alone wouldn't
        reliably signal that (an older baseline file can have an older mtime than the
        currently-published render)."""
        if mode == "absolute":
            return self._settings_signature({})
        return self._settings_signature({"baseline_year": resolve_baseline_year(self.settings)})

    def run(self, max_hours=None):
        # max_hours is a no-op here -- GHG renders once per cycle per (species, mode),
        # not per forecast hour. Accepted only so layer_builder's dispatch can call
        # every TASK_CLASSES entry's run() the same way.
        current_nc = camsforecast_cache_path(self.workdir)
        if not os.path.exists(current_nc):
            logger.info(
                "Greenhouse gases: CAMS forecast cache not present yet "
                "(data collector hasn't fetched it); skipping."
            )
            return

        baseline_year = resolve_baseline_year(self.settings)
        egg4_nc = egg4_baseline_cache_path(self.workdir, baseline_year)
        egg4_available = os.path.exists(egg4_nc)

        for species in SPECIES:
            for mode in ("absolute", "anomaly"):
                if mode == "anomaly" and not egg4_available:
                    logger.debug(
                        f"Greenhouse gases: EGG4 baseline {baseline_year} not present "
                        f"yet; skipping {species} anomaly."
                    )
                    continue

                out = self._output_path_for(species, mode)
                sources = [current_nc] + ([egg4_nc] if mode == "anomaly" else [])
                sig = self._mode_settings_signature(species, mode)
                fresh = self._is_render_fresh(out, sources, sig)
                if not fresh:
                    logger.info(f"Generating greenhouse gases {species} {mode} texture...")
                    self.plot(
                        species, mode, current_nc, egg4_nc if mode == "anomaly" else None, out
                    )
                    self._write_render_signature(out, sig)

        # Publish whichever (species, mode) is currently configured -- unconditionally
        # of whether it needed re-rendering above, same as SSTUpdater.run(). Skipped
        # only if the configured mode is anomaly and the baseline isn't cached yet
        # (nothing was rendered for it this cycle or any prior one).
        if self.mode == "anomaly" and not egg4_available:
            return

        self._publish_variant(self._output_path_for(self.species, self.mode))
