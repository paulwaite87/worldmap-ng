"""Standalone GFS data helpers, shared by the gfs_collector and (later) the tasks.

These mirror the byte-range / index / baseline logic that currently lives as methods
on tasks.common.Updater, but as free functions that don't need a full Updater instance
or a MapData. The collector downloads whole *hours* of data (the union of every layer's
GRIB targets in one hit) and stashes them in the DB; tasks then read from the DB instead
of each doing their own download.
"""

import logging
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

# Union of every GFS *atmospheric* layer's index targets — all served by the single
# gfs.tRUNz.pgrb2.0p25.fFFF file, so one ranged download per hour feeds all of them.
#   isobars       :PRMSL:mean sea level:
#   precipitation :PRATE:surface:
#   ozone         :TOZNE:
#   pwat          :PWAT:entire atmosphere (considered as a single layer):
#   stormwatch    :CAPE:surface: / :CIN:surface:
#   temperature   :TMP:2 m above ground:
#   wind          :UGRD:/:VGRD:10 m above ground:
#   jetstream     :UGRD:/:VGRD:250 mb: (jet-core pressure level)
ATMOS_TARGETS = [
    ":PRMSL:mean sea level:",
    ":PRATE:surface:",
    ":TOZNE:",
    ":PWAT:",
    ":CAPE:surface:",
    ":CIN:surface:",
    ":TMP:2 m above ground:",
    ":RH:2 m above ground:",
    ":UGRD:10 m above ground:",
    ":VGRD:10 m above ground:",
    ":UGRD:250 mb:",
    ":VGRD:250 mb:",
]

NOMADS_GFS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"


def build_atmos_url(base_url, date_str, run, fhour):
    return (
        f"{base_url}/gfs.{date_str}/{run}/atmos/"
        f"gfs.t{run}z.pgrb2.0p25.f{int(fhour):03d}"
    )


def build_wave_url(base_url, date_str, run, fhour):
    return (
        f"{base_url}/gfs.{date_str}/{run}/wave/gridded/"
        f"gfswave.t{run}z.global.0p25.f{int(fhour):03d}.grib2"
    )


def resolve_gfs_baseline(base_url=NOMADS_GFS_BASE, search_days=3):
    """Find the newest available GFS run by probing the lightweight f000 .idx sidecar.

    Returns {date_str, date_str_Y_M_D, run, timestamp(utc)} or None if nothing answered.
    This is the un-offset twin of Updater.get_gfs_state's baseline step — the collector
    wants data from 'now' forward, with no per-layer forecast_hour offset.
    """
    now = datetime.now(timezone.utc)
    for day_offset in range(search_days):
        target_date = now - timedelta(days=day_offset)
        date_str = target_date.strftime("%Y%m%d")
        for run in ["18", "12", "06", "00"]:
            url = (
                f"{base_url}/gfs.{date_str}/{run}/atmos/gfs.t{run}z.pgrb2.0p25.f000.idx"
            )
            try:
                if requests.head(url, timeout=5).status_code == 200:
                    ts = target_date.replace(
                        hour=int(run), minute=0, second=0, microsecond=0
                    )
                    logger.debug(f"GFS baseline: {date_str} {run}Z")
                    return {
                        "date_str": date_str,
                        "date_str_Y_M_D": target_date.strftime("%Y-%m-%d"),
                        "run": run,
                        "timestamp": ts,
                    }
            except requests.RequestException:
                continue
    return None


def resolve_gfs_baseline_with_coverage(
    base_url=NOMADS_GFS_BASE, cache_hours=24, search_days=3
):
    """Find the newest GFS run that can supply a FULL window of `cache_hours` forecast
    hours measured from 'now'.

    The plain resolver picks the newest run whose f000 exists — but GFS publishes its
    forecast hours incrementally over a few hours after a run starts, so a freshly-started
    run (f000 present) may not yet have the later hours we need to cover now..now+cache_hours.
    During that publish window we'd otherwise acquire a truncated range.

    This resolver instead, for each candidate run (newest first: today 18/12/06/00, then
    previous days), computes the highest forecast hour the window needs
    (fhour_0 + cache_hours - 1, where fhour_0 = hours since the run = the hour valid 'now')
    and probes whether THAT hour's .idx exists. GFS publishes hours in order, so if the top
    hour is present the whole window is. The first run that passes is used; otherwise we
    fall back to the previous run (00 -> previous-day 18 -> 12 -> 06 ...), which is fully
    published. Returns the same dict shape as resolve_gfs_baseline (+ "fhour_0"), or None.
    """
    now = datetime.now(timezone.utc)
    for day_offset in range(search_days):
        target_date = now - timedelta(days=day_offset)
        date_str = target_date.strftime("%Y%m%d")
        for run in ["18", "12", "06", "00"]:
            ts = target_date.replace(hour=int(run), minute=0, second=0, microsecond=0)
            if ts > now:
                continue  # a run in the future hasn't happened yet
            fhour_0 = max(0, int(round((now - ts).total_seconds() / 3600.0)))
            top_hour = fhour_0 + cache_hours - 1
            url = (
                f"{base_url}/gfs.{date_str}/{run}/atmos/"
                f"gfs.t{run}z.pgrb2.0p25.f{top_hour:03d}.idx"
            )
            try:
                if requests.head(url, timeout=5).status_code == 200:
                    logger.debug(
                        f"GFS baseline (full coverage): {date_str} {run}Z "
                        f"covers f{fhour_0:03d}..f{top_hour:03d}"
                    )
                    return {
                        "date_str": date_str,
                        "date_str_Y_M_D": target_date.strftime("%Y-%m-%d"),
                        "run": run,
                        "timestamp": ts,
                        "fhour_0": fhour_0,
                    }
            except requests.RequestException:
                continue
            logger.debug(
                f"GFS {date_str} {run}Z lacks f{top_hour:03d} (still publishing); "
                f"falling back to previous run."
            )
    # Nothing could supply a full window — fall back to the plain newest-available run so
    # we at least acquire what exists rather than nothing.
    logger.warning(
        "No GFS run can supply a full %dh window yet; using newest available run.",
        cache_hours,
    )
    return resolve_gfs_baseline(base_url, search_days=search_days)


def gfs_index_ranges(grib_url, targets, timeout=30):
    """Resolve (start, end) byte ranges for each target from the .idx sidecar.

    Returns a list of ranges (possibly shorter than `targets` if the freshest hour's
    sidecar hasn't fully populated yet — NOMADS often lags). Raises on network error.
    """
    if not targets:
        return []
    r = requests.get(grib_url + ".idx", timeout=timeout)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    ranges = []
    for target in targets:
        for i, line in enumerate(lines):
            if target in line:
                start = int(line.split(":")[1])
                end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else -1
                ranges.append((start, end))
                break
    return ranges


def download_byte_ranges(url, ranges, timeout=120):
    """Download the given byte ranges and return the concatenated bytes (in memory)."""
    out = bytearray()
    for start, end in ranges:
        hdr = {"Range": f"bytes={start}-" if end < 0 else f"bytes={start}-{end}"}
        r = requests.get(url, headers=hdr, timeout=timeout, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            out += chunk
    return bytes(out)


def download_whole(url, timeout=120, headers=None):
    """Download an entire file and return its bytes (for products like the wave GRIB).

    `headers` (e.g. a browser User-Agent) is passed through unchanged to requests.get
    for a source that WAF-blocks the default python-requests UA -- see
    ensure_gumbel_fit_cached (lib/flood_risk.py), confirmed live against ETH's
    research-collection host, which 403s a bare non-browser UA."""
    r = requests.get(url, timeout=timeout, stream=True, headers=headers)
    r.raise_for_status()
    out = bytearray()
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        out += chunk
    return bytes(out)


def remote_exists(url, timeout=10):
    try:
        return requests.head(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False