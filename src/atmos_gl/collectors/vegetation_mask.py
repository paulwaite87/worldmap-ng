#!/usr/bin/env python3
"""Keeps the Fire Risk layer's burnable-vegetation mask (lib/vegetation_mask.py)
current -- see issue #390.

Unlike a plain "fetch once forever" cache (this project's coastline.py GSHHG
pattern), this is a real periodic collector: land cover genuinely changes over time
(a large wildfire, deforestation), and the mask should eventually reflect that.
has_new_data() checks Zenodo's "versions/latest" API (a single cheap JSON GET)
against the last-downloaded version id, and only re-downloads the ~130MB GeoTIFF
when a newer annual release has actually been published. Annual is the real
freshness ceiling here -- this Zenodo mirror (like the underlying official MCD12Q1
product itself) only publishes a new classification once a year; a wildfire that
happens today won't be reflected in the mask until next year's release lands.
"""
import logging

from atmos_gl.collectors.base import CollectorBase
from atmos_gl.lib.vegetation_mask import (
    ZENODO_RECORD_HTML_URL,
    cached_version_id,
    download_landcover_geotiff,
    fetch_latest_zenodo_version,
    find_landcover_asset,
    save_cached_version_id,
    vegetation_mask_geotiff_cache_path,
)

logger = logging.getLogger(__name__)


class VegetationMaskCollector(CollectorBase):
    """Shares the Fires channel's enable gate (channel_key) since this mask has no
    purpose if Fire Risk/Fires isn't running; settings_section "fires" -- no
    independent Show/Config toggle of its own (see issue #390's "Config / UI
    surface" decision)."""

    section = "vegetation_mask"
    settings_section = "fires"
    channel_key = "fires"
    display_label = "MODIS Land Cover (Vegetation Mask)"

    def source_url(self) -> str | None:
        """Overridden: hardcoded Zenodo record, not a data_collector.datasources
        entry -- same "no config datasource, one real endpoint" convention as
        FloodRiskHistoricalCollector's JRC_BASE_URL."""
        return ZENODO_RECORD_HTML_URL

    def has_new_data(self) -> bool:
        """Stashes the checked version on self._latest_version for collect() to
        reuse, same "has_new_data() does the real check, collect() reuses it"
        contract as FloodRiskLiveCollector.has_new_data(). On a failed check,
        returns False (skip this cycle, retry next) rather than the CollectorBase
        default of "proceed anyway" -- collect() would just re-run this same check
        and hit the same failure, so there's nothing useful to attempt."""
        try:
            self._latest_version = fetch_latest_zenodo_version()
        except Exception as e:
            logger.debug(f"{self.section}: version check failed ({e}); will retry.")
            self._latest_version = None
            return False
        return self._latest_version.get("id") != cached_version_id(self.workdir)

    def collect(self) -> None:
        version = getattr(self, "_latest_version", None)
        if version is None:
            # has_new_data() wasn't run first (e.g. a direct call in a test) --
            # refetch rather than assume there's nothing to do.
            try:
                version = fetch_latest_zenodo_version()
            except Exception as e:
                logger.warning(f"{self.section}: version check failed ({e}); skipping.")
                return

        asset = find_landcover_asset(version)
        if asset is None:
            logger.warning(
                f"{self.section}: no land-cover asset found in latest Zenodo "
                f"version {version.get('id')}; skipping."
            )
            return

        dest = vegetation_mask_geotiff_cache_path(self.workdir)
        try:
            download_landcover_geotiff(asset["links"]["self"], dest)
        except Exception as e:
            logger.warning(f"{self.section}: download failed ({e}); will retry next cycle.")
            return

        save_cached_version_id(self.workdir, version.get("id"))
        logger.info(
            f"{self.section}: land-cover mask source updated to Zenodo version "
            f"{version.get('id')} ({asset['key']})."
        )
