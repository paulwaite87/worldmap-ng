#!/usr/bin/env python3
"""GDELT Event Database 2.0 -> database ("World Events" layer -- see CONTEXT.md).
Real-time export files, tab-delimited, no header, 61 fields (see _COL_* below, field
positions verified against a live export file rather than assumed from GDELT's own
published codebook alone, which is easy to miscount from: EventBaseCode and the two
Actor*Geo_ADM2Code fields sit between more commonly-cited field names).

Filtered to a curated, high-signal CAMEO code allowlist, not broad root bands --
GDELT logs on the order of hundreds of thousands of events/day across 300+ codes,
nowhere near all of it is a "world event" in the sense this layer means. Four
categories:
  - Explosion: 183, 1831-1833 (suicide/car/roadside bombing)
  - Warfare: 190-196 (conventional force through ceasefire violation)
  - Targeted/mass violence: 181, 185, 186, 200-204, 2041, 2042
  - Diplomacy: root 04 (040-046) AND an Actor1Name/Actor2Name match against a curated
    organization list -- root 04 alone is too generic (any two officials on a routine
    call would qualify); see _DIPLOMACY_ORGS.

Explosion/Warfare/Targeted-violence additionally require at least one actor to resolve
to a real state/military/organized-group entity (a non-blank Actor1/Actor2 CountryCode
or Type1-3Code -- see _has_state_actor()), the same actor-vetting principle Diplomacy
already applies via _DIPLOMACY_ORGS. GDELT's NLP event-coder matches on bare verb
phrases and will code figurative "battle"/"fight"/"war" language -- e.g. a film-casting
article about "the battle to become the next James Bond" -- into these conflict bands;
such an article's actors don't resolve to anything in CAMEO's country/military/group
dictionaries the way a real conflict's actors do, so requiring that resolution filters
the false positive at the source instead of guessing at tone/mention thresholds.

has_new_data() diffs lastupdate.txt's named export file against the last one actually
processed, so an unchanged 15-min window costs only a small text fetch. Backfill is
coverage-based, not empty-table-gated: every collect() compares the oldest stored
event against now - backfill_days and walks masterfilelist.txt for whatever gap
remains, running each missing file through the same fetch/parse/filter/upsert path a
normal cycle uses -- a collector that was down for a day self-heals the gap on its
next successful cycle, rather than only ever seeding once.
"""
import io
import logging
import re
import zipfile
from datetime import datetime, timedelta, timezone

import requests

from atmos_gl.collectors.base import CollectorBase
from atmos_gl.db.world_event_adapter import WorldEventAdapter

logger = logging.getLogger(__name__)

# 0-indexed column positions in GDELT 2.0's tab-delimited export CSV.
_COL_GLOBALEVENTID = 0
_COL_ACTOR1NAME = 6
_COL_ACTOR1COUNTRYCODE = 7
_COL_ACTOR1TYPE1CODE = 12
_COL_ACTOR1TYPE2CODE = 13
_COL_ACTOR1TYPE3CODE = 14
_COL_ACTOR2NAME = 16
_COL_ACTOR2COUNTRYCODE = 17
_COL_ACTOR2TYPE1CODE = 22
_COL_ACTOR2TYPE2CODE = 23
_COL_ACTOR2TYPE3CODE = 24
_COL_EVENTCODE = 26
_COL_GOLDSTEIN = 30
_COL_NUM_MENTIONS = 31
_COL_NUM_SOURCES = 32
_COL_AVG_TONE = 34
_COL_ACTIONGEO_FULLNAME = 52
_COL_ACTIONGEO_LAT = 56
_COL_ACTIONGEO_LONG = 57
_COL_DATEADDED = 59
_COL_SOURCEURL = 60
_MIN_COLS = 61

_EXPLOSION_CODES = {"183", "1831", "1832", "1833"}
_WARFARE_CODES = {"190", "191", "192", "193", "194", "195", "196"}
_TARGETED_VIOLENCE_CODES = {
    "181", "185", "186", "200", "201", "202", "203", "204", "2041", "2042",
}
_DIPLOMACY_EVENT_CODES = {"040", "041", "042", "043", "044", "045", "046"}

# Curated, not exhaustive -- high-profile multinational/summit bodies only, so a
# root-04 event between two ordinary officials doesn't qualify as a World Event (see
# module docstring). Matched case-insensitively as a substring against Actor1Name/
# Actor2Name, which GDELT already resolves to plain organization-name text.
_DIPLOMACY_ORGS = (
    "NATO", "UNITED NATIONS", "G7", "G8", "G20", "EUROPEAN UNION", "ASEAN", "OPEC",
    "AFRICAN UNION", "ARAB LEAGUE", "WORLD ECONOMIC FORUM",
)

_EXPORT_FILE_RE = re.compile(r"(\d{14})\.export\.CSV\.zip$")


def _has_state_actor(*codes: str | None) -> bool:
    """True when at least one Actor1/Actor2 CountryCode or Type1-3Code field resolved
    to something. GDELT's CAMEO actor dictionary only populates these when the raw
    actor text matched a known country, government, military, or organized-group
    pattern -- an unresolvable proper noun (e.g. a film-casting article's "Pierce
    Brosnan") leaves them all blank even though Actor1Name/Actor2Name still carry the
    raw text. See module docstring for why this gates the conflict categories."""
    return any((c or "").strip() for c in codes)


def _classify(
    event_code: str,
    actor1_name: str | None,
    actor2_name: str | None,
    has_state_actor: bool = False,
) -> str | None:
    if event_code in _EXPLOSION_CODES:
        return "explosion" if has_state_actor else None
    if event_code in _WARFARE_CODES:
        return "warfare" if has_state_actor else None
    if event_code in _TARGETED_VIOLENCE_CODES:
        return "targeted_violence" if has_state_actor else None
    if event_code in _DIPLOMACY_EVENT_CODES:
        names = f"{actor1_name or ''} {actor2_name or ''}".upper()
        if any(org in names for org in _DIPLOMACY_ORGS):
            return "diplomacy"
    return None


def _parse_export_rows(csv_text: str) -> list[dict]:
    """Parses one GDELT export CSV's text into row dicts ready for
    WorldEventAdapter.upsert_events(), pre-filtered to the curated category allowlist
    (see module docstring). Deliberately does NOT apply the min_mentions floor --
    that's a settings-tunable threshold the caller applies, not baked into parsing."""
    rows = []
    for line in csv_text.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < _MIN_COLS:
            continue

        event_code = parts[_COL_EVENTCODE].strip()
        actor1_name = parts[_COL_ACTOR1NAME].strip() or None
        actor2_name = parts[_COL_ACTOR2NAME].strip() or None
        has_state_actor = _has_state_actor(
            parts[_COL_ACTOR1COUNTRYCODE], parts[_COL_ACTOR1TYPE1CODE],
            parts[_COL_ACTOR1TYPE2CODE], parts[_COL_ACTOR1TYPE3CODE],
            parts[_COL_ACTOR2COUNTRYCODE], parts[_COL_ACTOR2TYPE1CODE],
            parts[_COL_ACTOR2TYPE2CODE], parts[_COL_ACTOR2TYPE3CODE],
        )
        category = _classify(event_code, actor1_name, actor2_name, has_state_actor)
        if category is None:
            continue

        lat_str = parts[_COL_ACTIONGEO_LAT].strip()
        lon_str = parts[_COL_ACTIONGEO_LONG].strip()
        if not lat_str or not lon_str:
            continue

        try:
            lat, lon = float(lat_str), float(lon_str)
            num_mentions = int(parts[_COL_NUM_MENTIONS].strip() or 0)
            num_sources = int(parts[_COL_NUM_SOURCES].strip() or 0)
            goldstein_raw = parts[_COL_GOLDSTEIN].strip()
            goldstein = float(goldstein_raw) if goldstein_raw else None
            tone_raw = parts[_COL_AVG_TONE].strip()
            avg_tone = float(tone_raw) if tone_raw else None
            event_date = datetime.strptime(
                parts[_COL_DATEADDED].strip(), "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue

        rows.append(
            {
                "id": parts[_COL_GLOBALEVENTID].strip(),
                "category": category,
                "event_code": event_code,
                "actor1_name": actor1_name,
                "actor2_name": actor2_name,
                "action_geo_full_name": parts[_COL_ACTIONGEO_FULLNAME].strip() or None,
                "lat": lat,
                "lon": lon,
                "event_date": event_date.isoformat(),
                "num_mentions": num_mentions,
                "num_sources": num_sources,
                "goldstein_scale": goldstein,
                "avg_tone": avg_tone,
                "source_url": parts[_COL_SOURCEURL].strip() or None,
            }
        )
    return rows


def _fetch_export_csv(url: str) -> str:
    """Downloads one GDELT export.CSV.zip and returns its single member's decoded
    text. GDELT's export files are latin-1 (confirmed against a live file: several
    actor/place names carry raw high-byte characters that aren't valid UTF-8)."""
    r = requests.get(url, timeout=30, headers={"User-Agent": "AtmosGL-Collector/1.0"})
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        return zf.read(name).decode("latin-1")


def _export_url_from_lastupdate(text: str) -> str | None:
    for line in text.splitlines():
        if ".export.CSV.zip" in line:
            parts = line.split()
            if parts:
                return parts[-1]
    return None


class WorldEventsCollector(CollectorBase):
    section = "world_events"
    channel_key = "world_events"
    datasource_key = "world_events"

    def __init__(self, config):
        super().__init__(config)
        self.world_event_adapter = WorldEventAdapter()

    def _lastupdate_url(self) -> str:
        return f"{self.datasource_url('world_events').rstrip('/')}/lastupdate.txt"

    def _masterfilelist_url(self) -> str:
        return f"{self.datasource_url('world_events').rstrip('/')}/masterfilelist.txt"

    def has_new_data(self) -> bool:
        """Cheap freshness check: lastupdate.txt always names the current export file
        -- compare against the last one actually processed (cached by URL, like
        CollectorBase's own HEAD-based ETag cache) rather than re-downloading/parsing
        on every driver poll faster than GDELT's own 15-min cadence."""
        lastupdate_url = self._lastupdate_url()
        try:
            r = requests.get(
                lastupdate_url, timeout=10, headers={"User-Agent": "AtmosGL-Collector/1.0"}
            )
            r.raise_for_status()
            url = _export_url_from_lastupdate(r.text)
        except Exception as e:
            logger.debug(f"World Events: lastupdate.txt fetch failed: {e}")
            return True  # can't tell -> collect anyway, safe fallback

        if not url:
            return True
        if self._etag_cache.get(lastupdate_url) == url:
            return False
        return True

    def _backfill_gap(self, backfill_days: int, min_mentions: int) -> None:
        """Walks masterfilelist.txt for whatever part of the backfill_days window
        isn't covered yet by the oldest stored event, running each missing export
        file through the same fetch/parse/filter/upsert path collect() uses.

        masterfilelist.txt is chronologically ascending back to 2015, so the very
        first backfill (an empty table) scans from the start of that huge file up to
        where the window begins -- a one-time cost accepted for simplicity rather
        than an HTTP Range-based tail-seek; every subsequent cycle is a fast no-op
        once expiry_days' worth of retained data already exceeds backfill_days.
        """
        now = datetime.now(timezone.utc)
        target_start = now - timedelta(days=backfill_days)

        oldest = self.world_event_adapter.oldest_event_date()
        if oldest is not None and oldest <= target_start:
            return  # already have full coverage

        gap_end = oldest if oldest is not None else now
        logger.info(
            f"World Events: backfilling coverage from {target_start.isoformat()} "
            f"to {gap_end.isoformat()}."
        )

        try:
            r = requests.get(
                self._masterfilelist_url(), timeout=60,
                headers={"User-Agent": "AtmosGL-Collector/1.0"}, stream=True,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f"World Events: masterfilelist.txt fetch failed: {e}")
            return

        urls = []
        for line in r.iter_lines(decode_unicode=True):
            if not line or ".export.CSV.zip" not in line:
                continue
            parts = line.split()
            if not parts:
                continue
            m = _EXPORT_FILE_RE.search(parts[-1])
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if ts < target_start:
                continue
            if ts >= gap_end:
                break  # ascending order -- nothing further is needed
            urls.append(parts[-1])

        upserted = 0
        for url in urls:
            try:
                rows = _parse_export_rows(_fetch_export_csv(url))
                rows = [row for row in rows if row["num_mentions"] >= min_mentions]
                self.world_event_adapter.upsert_events(rows)
                upserted += len(rows)
            except Exception as e:
                logger.warning(f"World Events: backfill file {url} failed, skipping: {e}")
                continue

        logger.info(
            f"World Events: backfill complete, upserted {upserted} event(s) "
            f"across {len(urls)} file(s)."
        )

    def collect(self) -> None:
        min_mentions = int(self.settings.get("min_mentions", 10))
        backfill_days = int(self.settings.get("backfill_days", 3))

        self._backfill_gap(backfill_days, min_mentions)

        lastupdate_url = self._lastupdate_url()
        try:
            r = requests.get(
                lastupdate_url, timeout=10, headers={"User-Agent": "AtmosGL-Collector/1.0"}
            )
            r.raise_for_status()
            url = _export_url_from_lastupdate(r.text)
        except Exception as e:
            logger.error(f"World Events: lastupdate.txt fetch failed: {e}")
            return

        if not url:
            logger.warning("World Events: lastupdate.txt has no export.CSV.zip entry; skipping.")
            return

        rows = _parse_export_rows(_fetch_export_csv(url))
        rows = [row for row in rows if row["num_mentions"] >= min_mentions]
        self.world_event_adapter.upsert_events(rows)
        self._etag_cache[lastupdate_url] = url
        logger.info(f"World Events: upserted {len(rows)} event(s) from {url}.")
