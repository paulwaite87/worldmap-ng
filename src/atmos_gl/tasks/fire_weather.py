#!/usr/bin/env python3
"""Fire Weather Index heatmap -- renders under the SAME "fires" config section/toggle
the FIRMS hotspot layer (routes/fires.py, collectors/fires.py) uses, so the Show tab
needs only one "Wildfires" checkbox, not two.

FireWeatherUpdater reuses ScalarFieldUpdater's plot()/_resolve_cmap()/_write_legend_key()/
run() entirely unchanged (see tasks/scalar_field.py) -- the ONLY reason this isn't a
plain SPECS entry there is that ScalarFieldUpdater.__init__ ties self.section to
spec.product (Updater.__init__(config, spec.product, map_data)), and this task
deliberately needs them to differ: config section "fires" (shared with the FIRMS
collector/route), fieldstore product "fire_weather" (kept distinct from the unrelated
`fires` DB table so logs/code aren't ambiguous about which "fires" is meant). Overriding
__init__ to call Updater.__init__ directly is the one place that decouples them.

Data source: fire_weather_data_unpack (lib/unpack.py), registered in ATMOS_UNPACKERS,
computes the Fosberg Fire Weather Index from GFS temperature/humidity/wind -- see that
module for the formula. Collected automatically by GfsAtmosCollector like every other
atmos product; no collector changes needed for this layer.
"""
import numpy as np

from atmos_gl.lib.coastline import LandMaskCache
from atmos_gl.lib.vegetation_mask import VegetationMaskCache
from .common import Updater
from .scalar_field import ScalarFieldUpdater, ScalarFieldSpec

# A single fixed danger ramp (pale yellow at the threshold -> deep red at vmax, same
# visual family as YlOrRd/stormwatch) -- no palette CHOICE is exposed in config (no
# ("fires", "palette") FIELD_SPECS entry), but _resolve_cmap's threshold-rendering branch
# (see scalar_field.py) unconditionally does spec.palettes.get(...), so a real one-entry
# dict is required even though there's only ever one option.
_FIRE_WEATHER_PALETTE = {
    "default": [(1.0, 0.95, 0.6), (1.0, 0.55, 0.1), (0.6, 0.0, 0.0)],
}

FIRE_WEATHER_SPEC = ScalarFieldSpec(
    product="fire_weather",
    vmin=0.0,
    vmax=100.0,
    extend="max",  # rare extreme conditions can exceed 100
    ticks=[0, 20, 40, 60, 80, 100],
    title="Fire Risk",
    # "Critical zone" rendering (see scalar_field.py's _threshold_colormap): only shade
    # elevated-risk areas, same mechanism ozone/pwat already use for "don't colourise
    # the whole globe, just the areas that matter" -- most of the planet has near-zero
    # fire risk at any given moment and shouldn't be tinted.
    threshold_setting="min_risk_display",
    threshold_default=25.0,
    focus="above",  # worse (more dangerous) toward vmax, like pwat
    flat_color=(0.0, 0.0, 0.0, 0.0),  # fully transparent below the threshold
    palette_default="default",
    palettes=_FIRE_WEATHER_PALETTE,
)


class FireWeatherUpdater(ScalarFieldUpdater):
    def __init__(self, config, map_data):
        Updater.__init__(self, config, "fires", map_data)
        self.spec = FIRE_WEATHER_SPEC
        self.level_of_detail = int(self.settings.get("level_of_detail", 1))
        self.lod_desc = None
        self.per_hour_outputs = [".png", "_data.png"]
        self.status_product = "fire_weather"
        # Land + vegetation masking (issue #390): Fire Risk is the one
        # ScalarFieldUpdater spec that isn't meaningful everywhere on the globe --
        # unlike temperature/ozone/CAPE/pwat, fire danger doesn't exist over open
        # ocean or unvegetated land, however dry and windy conditions there are.
        self._land_mask_cache = LandMaskCache("FireWeather")
        self._vegetation_mask_cache = VegetationMaskCache("FireWeather", self.workdir)

    def _mask_values(self, values, lats, lons):
        """ANDs land + burnable-vegetation together; each degrades independently
        if its underlying data isn't available yet (first boot) or fails to load,
        and the layer never renders fully blank just because one mask is missing
        -- see issue #390's "Fallback / degradation behavior" decision.

        Cache key includes each grid's latitude endpoints, not just its shape:
        this is called once against the native-resolution grid and once against
        the LOD-regridded grid every render, and those two grids can coincidentally
        share a shape (though never the same endpoints/ordering) -- keying on
        shape alone would let one grid's cached mask get silently reused for the
        other, which is wrong in both content and latitude ordering (both caches'
        `shape`/`key` parameter is just an opaque cache key, never compared
        against the mask's actual shape -- see VegetationMaskCache's docstring)."""
        values = np.asarray(values, dtype=np.float64)
        shape = values.shape
        key = (shape, float(lats[0]), float(lats[-1]))
        land = self._land_mask_cache.get(lats, lons, key)
        burnable = self._vegetation_mask_cache.get(lats, lons, key)
        if land is None and burnable is None:
            return values
        keep = np.ones(shape, dtype=bool)
        if land is not None:
            keep &= land
        if burnable is not None:
            keep &= burnable
        return np.where(keep, values, np.nan)
