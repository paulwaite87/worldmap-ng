#!/usr/bin/env python3
"""Flood Risk layer collectors -- see issue #371 (original design) and its
follow-up grilling (Live mode's data-source pivot, below).

  FloodRiskLiveCollector       -- Live mode: NASA LANCE MODIS flood product
                                  ("Observed Current Inundation"), rebuilt from
                                  up to 287 10x10deg GeoTIFF tiles every cycle a
                                  tile changes or expires.
  FloodRiskHistoricalCollector -- Historical mode: JRC Global River Flood Hazard
                                  Maps (100-year return period), mosaicked once
                                  into a single global raster and cached forever.

Both share one settings_section ("flood_risk", holding the shared mode toggle) but
keep independent `section`/channel identities, same split as the greenhouse_gases
layer's forecast/baseline pair.

Live mode originally fetched Copernicus GloFAS's ensemble discharge FORECAST via
EWDS, classified against ETH's Gumbel-fit return-period thresholds -- see git
history (PRs #383/#384) for that design. It was abandoned entirely rather than
further patched, after:
  1. A real OOM crash (fixed in PR #383: an unnecessary float64 upcast of a
     ~4.3GB ensemble array), and
  2. A SECOND, distinct OOM discovered after that fix and after PR #384's
     per-leadtime-hour resumable-fetch split (which only addressed network
     flakiness, not memory) -- a single leadtime hour's raw ensemble array alone
     (~4.3GB) plus overhead still exceeded this 11GB-RAM host's headroom.
  3. Severe, independently-confirmed network flakiness against ECMWF/
     Copernicus's shared object-store backend (the same symptom was also seen
     for CAMS, a different dataset on the same backend) -- not fixable from this
     app's side.
On top of those infrastructure problems, a forecast product was never a great
match for a "Live" mode sitting next to Historical's hazard-potential map anyway.
MODIS's flood product instead reports OBSERVED current inundation -- an
architecturally simpler integration (plain authenticated HTTPS GeoTIFF downloads,
no CDS/EWDS job-queue) as well as a better conceptual fit.
"""
import logging
import os
from datetime import datetime, timezone

import numpy as np

from atmos_gl.collectors.base import CollectorBase
from atmos_gl.lib.data_status import build_status, read_process_status
from atmos_gl.lib.flood_risk import (
    JRC_BASE_URL,
    LANCE_BASE_URL,
    build_jrc_mosaic_grid,
    cached_modis_flood_tiles,
    count_cached_jrc_tiles,
    ensure_jrc_tile_cached,
    ensure_jrc_tile_extents_cached,
    ensure_modis_flood_tile_cached,
    fetch_modis_flood_listing,
    jrc_hazard_mosaic_cache_path,
    jrc_tile_cache_path,
    load_jrc_tile_index,
    modis_flood_mosaic_cache_path,
    modis_flood_tile_bounds,
    modis_flood_tile_is_current,
    prune_stale_modis_flood_tiles,
    resample_jrc_tile_onto_grid,
    resample_modis_flood_tile_onto_grid,
    resolve_earthdata_token,
    save_jrc_hazard_mosaic,
    tile_dst_window,
)

logger = logging.getLogger(__name__)


class FloodRiskLiveCollector(CollectorBase):
    """Live mode: NASA LANCE MODIS 1-Day cloud-shadow-screened flood product
    ("Observed Current Inundation"), rebuilt into a full global mosaic every cycle
    has_new_data() finds at least one changed or newly-expired tile.

    Deliberately a CollectorBase subclass (file-cache family, driven by
    EventFeedDriver via collect_file_caches()), NOT FieldCollectorBase: there is no
    forecast-hour dimension here at all -- MODIS reports a single continuously-
    refreshed "current" state, not a run+leadtime series, so the per-(run_date,
    run_id, fhour) field-catalog model GFS/RTOFS use doesn't apply. This mirrors
    FloodRiskHistoricalCollector's own storage shape (one cached global raster
    file) exactly, not the old GloFAS collector's.

    has_new_data() does the real remote check (a single cheap JSON listing
    request) and stashes it on `self._listing` for collect() to reuse -- safe
    because EventFeedDriver constructs a fresh instance per drive() call and
    always calls has_new_data() immediately before collect() on that same
    instance (driving.py's EventFeedDriver._drive_one). It also prunes any tile
    that's aged past staleness as a side effect (see prune_stale_modis_flood_tiles's
    docstring) -- an "Observed CURRENT Inundation" claim must not keep rendering
    days-old flood pixels through a LANCE outage.

    collect() then downloads only the tiles the listing says changed (tiles that
    fail to refresh this cycle simply keep contributing their last-known-good
    cached content -- same resilience JRC's per-tile cache already provides for
    Historical mode) and rebuilds the ENTIRE mosaic from whatever's cached,
    streaming one tile's small GeoTIFF into the shared grid at a time -- unlike
    the old GloFAS design, a MODIS tile is tiny (a single 8-bit band, tens of KB
    compressed), so there is no OOM risk here even rebuilding all 287 tiles
    every time.
    """

    section = "flood_risk"
    settings_section = "flood_risk"
    channel_key = "flood_risk_live"
    display_label = "NASA MODIS Flood (Live)"

    def source_url(self) -> str | None:
        """Overridden: hardcoded LANCE endpoint, not a data_collector.datasources
        entry -- same "no config datasource, one real endpoint" convention as
        FloodRiskHistoricalCollector's own JRC_BASE_URL."""
        return LANCE_BASE_URL

    def has_new_data(self) -> bool:
        pruned = prune_stale_modis_flood_tiles()
        token = resolve_earthdata_token(self.channel_key)
        if token is None:
            self._listing = []
            return pruned
        try:
            self._listing = fetch_modis_flood_listing(datetime.now(timezone.utc), token)
        except Exception as e:
            logger.debug(f"{self.channel_key}: tile listing unavailable ({e}).")
            self._listing = []
            return pruned
        changed = any(not modis_flood_tile_is_current(t) for t in self._listing)
        if not changed and not pruned:
            logger.debug(f"{self.channel_key}: no changed or expired tiles; skipping.")
        return changed or pruned

    def collect(self) -> None:
        token = resolve_earthdata_token(self.channel_key)
        if token is None:
            return

        listing = getattr(self, "_listing", None)
        if listing is None:
            # has_new_data() wasn't run first (e.g. a direct call in a test) --
            # refetch rather than assume there's nothing to do.
            try:
                listing = fetch_modis_flood_listing(datetime.now(timezone.utc), token)
            except Exception as e:
                logger.warning(f"{self.channel_key}: tile listing unavailable ({e}); skipping.")
                return

        downloaded = 0
        for tile in listing:
            if modis_flood_tile_is_current(tile):
                continue
            try:
                ensure_modis_flood_tile_cached(tile, token)
                downloaded += 1
            except Exception as e:
                logger.warning(
                    f"{self.channel_key}: tile h{tile['h']:02d}v{tile['v']:02d} "
                    f"unavailable this cycle ({e}); keeping previous version if any."
                )

        cached_tiles = cached_modis_flood_tiles()
        if not cached_tiles:
            logger.info(f"{self.channel_key}: no tiles cached yet; nothing to mosaic.")
            return

        lat, lon = build_jrc_mosaic_grid()
        mosaic = np.zeros((len(lat), len(lon)), dtype=np.uint8)
        for h, v, tile_path in cached_tiles:
            row0, row1, col0, col1 = tile_dst_window(modis_flood_tile_bounds(h, v))
            mosaic[row0:row1, col0:col1] = resample_modis_flood_tile_onto_grid(
                tile_path, lat[row0:row1], lon[col0:col1]
            )

        save_jrc_hazard_mosaic(modis_flood_mosaic_cache_path(self.workdir), mosaic, lat, lon)
        logger.info(
            f"{self.channel_key}: mosaic rebuilt ({len(cached_tiles)} tile(s) cached, "
            f"{downloaded} newly downloaded this cycle)."
        )


class FloodRiskHistoricalCollector(CollectorBase):
    """Historical mode: JRC Global River Flood Hazard Maps at the 100-year return
    period, mosaicked once into a single global raster and cached forever -- a
    fixed, terrain/floodplain-derived hazard classification, unlike Live mode's
    continuously-refreshed MODIS observation. CollectorBase (fetch-once style, like
    CamsEgg4BaselineCollector/GSHHG) is correct here: there is no time dimension at
    all to store per-forecast-hour.

    271 tiles (~515MB total, RP100 reclass variant only, see lib/flood_risk.py's
    module docstring) are downloaded across however many collect() cycles it
    takes -- ensure_jrc_tile_cached() skips tiles already on disk, so a partial
    pass just resumes next cycle rather than re-downloading from scratch. The
    final mosaic is only built and cached once ALL 271 tiles have downloaded
    successfully in one pass; a partial pass logs progress and returns, leaving
    the (nonexistent) mosaic cache file as the "not yet done" signal for the next
    cycle -- no separate "download complete" flag needed.
    """

    section = "flood_risk_historical"
    settings_section = "flood_risk"
    channel_key = "flood_risk_historical"
    display_label = "JRC Flood Hazard (Historical)"

    # collect() runs synchronously inside CollectorService.collect_once()'s single
    # sequential sweep (collectors/service.py) -- everything after this collector in
    # that sweep (event feeds, then GFS/RTOFS field ingestion), AND the
    # "data_collector" service heartbeat itself, all wait for collect() to return.
    # Downloading every remaining tile in one call can take long enough (network
    # latency x up to 271 tiles, ~515MB total -- ensure_jrc_tile_cached()'s own
    # docstring notes this host observed mid-transfer failures on a 271-tile batch)
    # to push that heartbeat past the Data Status page's dead threshold, which reads
    # as the WHOLE data_collector service being down even though it's just busy with
    # this one-time historical backfill. Capping actual NEW downloads per call
    # (already-cached tiles are free -- they don't count) bounds collect()'s
    # wall-clock time regardless of how many tiles remain, spreading the initial
    # backfill across is_stale()'s normal hourly cadence instead -- matching this
    # class's own docstring, which already claimed (but didn't enforce) that shape.
    _MAX_NEW_TILE_DOWNLOADS_PER_CYCLE = 30

    def source_url(self) -> str | None:
        """Overridden: hardcoded open-FTP source, not a data_collector.datasources
        entry -- same "no config datasource" convention as StormsCollector's own
        ATCF mirror URLs (see CollectorBase.datasource_key's docstring)."""
        return JRC_BASE_URL

    def collect(self) -> None:
        dest = jrc_hazard_mosaic_cache_path(self.workdir)
        if os.path.exists(dest):
            logger.debug(f"{self.section}: mosaic already cached; skipping.")
            return

        try:
            index_path = ensure_jrc_tile_extents_cached()
            tiles = load_jrc_tile_index(index_path)
        except Exception as e:
            logger.warning(f"{self.section}: tile index unavailable ({e}); skipping this cycle.")
            return

        lat, lon = build_jrc_mosaic_grid()
        mosaic = np.zeros((len(lat), len(lon)), dtype=np.uint8)

        cached_count = 0
        new_downloads = 0
        for tile in tiles:
            already_cached = os.path.exists(jrc_tile_cache_path(tile["id"], tile["name"]))
            if not already_cached and new_downloads >= self._MAX_NEW_TILE_DOWNLOADS_PER_CYCLE:
                logger.info(
                    f"{self.section}: per-cycle download budget "
                    f"({self._MAX_NEW_TILE_DOWNLOADS_PER_CYCLE} new tiles) reached "
                    f"({cached_count}/{len(tiles)} cached so far); will resume next cycle."
                )
                return

            try:
                tile_path = ensure_jrc_tile_cached(tile["id"], tile["name"])
            except Exception as e:
                logger.warning(
                    f"{self.section}: tile {tile['name']!r} unavailable this cycle "
                    f"({e}); will retry next cycle."
                )
                continue
            if not already_cached:
                new_downloads += 1
            cached_count += 1

            row0, row1, col0, col1 = tile_dst_window(tile["bounds"])
            mosaic[row0:row1, col0:col1] = resample_jrc_tile_onto_grid(
                tile_path, lat[row0:row1], lon[col0:col1]
            )

        if cached_count < len(tiles):
            logger.info(
                f"{self.section}: {cached_count}/{len(tiles)} tiles cached so far; "
                f"mosaic not yet complete."
            )
            return

        save_jrc_hazard_mosaic(dest, mosaic, lat, lon)
        logger.info(f"{self.section}: mosaic complete ({len(tiles)} tiles) -> {dest}")

    def data_status(self) -> dict:
        """Coverage-based, not time-decay -- same reasoning as
        CamsEgg4BaselineCollector.data_status(): this collector fetches once (across
        however many cycles the 271-tile download takes) then is permanently done,
        so a decaying-freshness formula would show perpetual staleness for a source
        working exactly as designed. Percent tracks tile-download progress until the
        mosaic itself is cached, at which point it's simply 100."""
        last_updated, last_error, status = read_process_status(
            self.process_status_adapter, self.section
        )
        mosaic_cached = os.path.exists(jrc_hazard_mosaic_cache_path(self.workdir))
        counts = count_cached_jrc_tiles()

        if mosaic_cached:
            percent = 100.0
            detail = last_error or "mosaic cached"
        elif counts:
            cached, total = counts
            percent = 100.0 * cached / total if total else 0.0
            detail = last_error or f"{cached}/{total} tiles cached"
        else:
            percent = 0.0
            detail = last_error or "tile index not yet fetched"

        return build_status(
            name=self.section,
            kind="collector",
            percent=percent,
            last_updated=last_updated,
            next_update=None,
            enabled=self.enabled,
            detail=detail,
            status=status,
        )
