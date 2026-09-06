#!/usr/bin/env python3
"""Flight Radar's data acquisition (issue #203/#215). Originally region-keyed
backend-proxy-and-push (docs/adr/0009, superseded by docs/adr/0010): pure geometry
helpers (circle_for_region_key, fetch_aircraft_near) plus a stateful RegionManager
driving a WebSocket-push route. RegionManager and its WS route (routes/flightradar.py)
were removed once AircraftCollector (collectors/aircraft.py) took over as adsb.lol's
sole consumer -- GlobalSampleScheduler below is what actually schedules sampling now.
The geometry helpers survive unchanged; AircraftCollector reuses circle_for_region_key
and fetch_aircraft_near exactly as RegionManager's poll loop did.
"""
import logging
import math

import aiohttp

logger = logging.getLogger("atmos_gl.lib.flight_radar")

ADSB_LOL_BASE = "https://api.adsb.lol/v2"

# The routeset endpoint lives at a different path prefix on the same host as
# ADSB_LOL_BASE (/api/0/routeset, not /v2/...) -- not derivable from ADSB_LOL_BASE by
# string surgery, so it's its own constant/datasource entry
# (data_collector.datasources.flightradar_routeset).
#
# Points at adsb.im, NOT api.adsb.lol, despite the constant name's "ADSB_LOL" prefix
# (kept for naming consistency with ADSB_LOL_BASE/ADSB_LOL_BASE-derived code, since
# both hosts run the identical open-source adsblol/api project). Verified live
# (2026-07-26): api.adsb.lol/api/0/routeset -- and even its own OPTIONS preflight,
# which the server source hardcodes to return 200 -- currently returns a bare 201
# with an empty body and no CORS headers, while every GET endpoint on that same host
# (including /api/0/airport/{icao}, the same router) works fine. adsb.im runs the
# same codebase and responds correctly with the exact schema this module expects.
# Safe to repoint back to api.adsb.lol via the datasources config entry alone if
# that gets fixed upstream -- no code change needed either way.
ADSB_LOL_ROUTESET_BASE = "https://adsb.im/api/0/routeset"

# adsblol/api's own server-side cap (src/adsb_api/utils/api_routes.py: a request with
# more than 100 planes gets rejected with a plain 400) -- verified against the actual
# server source, not a guess.
ROUTESET_BATCH_LIMIT = 100

# Grid cell size in degrees for the fine/hotspot tier -- also GlobalSampleScheduler's
# FINE_GRID_DEG, so an active viewer's hot cell lines up with what the frontend itself
# considers "the area in view." Not tuned against real adsb.lol traffic yet; ~5deg
# (~550km at the equator) is a starting guess in the same ballpark as the hot circle's
# own radius, left for empirical tuning during rollout like every other numeric
# constant in this feature.
GRID_DEG = 5.0

# adsb.lol query radius, nautical miles. adsb.lol never confirmed a max radius during
# research; ADSBExchange-family APIs typically cap around 250nm. 200nm is a starting
# guess -- large enough to reasonably cover a GRID_DEG cell from its center (a 5deg
# cell's corner is ~215nm from center), tuned empirically once live.
CIRCLE_RADIUS_NM = 200.0


def _cell(lon: float, lat: float, grid_deg: float) -> tuple[int, int]:
    return (math.floor(lon / grid_deg), math.floor(lat / grid_deg))


def circle_for_region_key(
    region_key: tuple[int, int], *, grid_deg: float = GRID_DEG, radius_nm: float = CIRCLE_RADIUS_NM,
) -> tuple[float, float, float]:
    """A region key (grid cell) -> the (lat, lon, radius_nm) circle queried for it,
    centered on the cell. One circle per region key -- doesn't perfectly cover every
    corner of the cell at every grid_deg/radius_nm combination; an accepted, tunable
    imprecision (see CIRCLE_RADIUS_NM's docstring)."""
    lon_idx, lat_idx = region_key
    lon = (lon_idx + 0.5) * grid_deg
    lat = (lat_idx + 0.5) * grid_deg
    return lat, lon, radius_nm


async def fetch_aircraft_near(
    session: aiohttp.ClientSession, lat: float, lon: float, radius_nm: float,
    *, base_url: str = ADSB_LOL_BASE, timeout: float = 10.0, report_status=None,
) -> list[dict] | None:
    """One adsb.lol point+radius query -> its `ac` (aircraft) list, or None on any
    failure (timeout, non-200 -- adsb.lol's free tier 429s far more readily than its
    documented behaviour suggests, malformed response). None is deliberately distinct
    from [] : a failed request must never crash the poll loop, but it also must never
    be reported to callers as "confirmed zero aircraft here" -- see
    GlobalSampleScheduler.record_result(), whose whole reason for accepting None is
    this distinction.

    base_url defaults to ADSB_LOL_BASE but is normally overridden by the caller with
    the configured data_collector.datasources.flightradar value (AircraftCollector) --
    same "URL lives in the shared datasources dict, not hardcoded" convention every
    other collector follows.

    report_status, if given, is called with the raw HTTP status code once a response
    is actually received (never called if the request raised before completing --
    a timeout/connection error, as opposed to a real rejection). Purely a side-channel
    for Data Status health reporting (see AircraftCollector._report_status()),
    independent of this function's own None-vs-[] success/failure contract -- a single
    rate-limited request shouldn't be conflated with "the fetch failed"."""
    url = f"{base_url}/lat/{lat}/lon/{lon}/dist/{radius_nm}"
    try:
        async with session.get(url, timeout=timeout) as resp:
            if report_status:
                report_status(resp.status)
            if resp.status != 200:
                logger.debug(f"adsb.lol {url} returned {resp.status}")
                return None
            data = await resp.json()
            return data.get("ac", []) or []
    except Exception as exc:
        logger.debug(f"adsb.lol fetch failed for {url}: {exc}")
        return None


async def fetch_routes(
    session: aiohttp.ClientSession, planes: list[dict],
    *, base_url: str = ADSB_LOL_ROUTESET_BASE, timeout: float = 10.0, report_status=None,
) -> dict[str, dict | None] | None:
    """Batch-resolves callsign -> route via adsb.lol's routeset endpoint (issue #215's
    route-lookup follow-on). `planes` is [{"callsign": str, "lat": float, "lng": float}, ...]
    (real current position, not a 0/0 placeholder -- it's what lets adsb.lol compute the
    "plausible" great-circle sanity check below); callers must keep each batch at or
    under ROUTESET_BATCH_LIMIT, the server's own hard cap.

    Returns {callsign: {"stops": [...], "plausible": bool} | None} for every callsign
    the server actually responded about -- None (whole-batch failure: timeout, non-200)
    on any request-level failure, the same None-vs-populated-dict distinction
    fetch_aircraft_near makes, so a rejected batch is never misread as "every callsign
    in it has no route". A per-callsign None inside the dict is the server's own
    confirmed no-match ("airport_codes": "unknown"), distinct from a callsign simply
    absent from the response (left out of the returned dict entirely, so the caller
    retries it rather than wrongly recording a confirmed non-match).

    Matched by the "callsign" field each response entry carries (adsblol/api's
    api_routeset echoes it back onto every entry it builds), not by array position --
    more robust than assuming response order mirrors request order.

    stops preserves the full `_airports` list in order (origin first, destination
    last, any technical/intermediate stop(s) kept in between) rather than collapsing to
    just origin/destination, per this feature's Q8 design decision."""
    if not planes:
        return {}
    body = {
        "planes": [
            {"callsign": p["callsign"], "lat": p.get("lat", 0.0), "lng": p.get("lng", 0.0)}
            for p in planes
        ]
    }
    try:
        async with session.post(base_url, json=body, timeout=timeout) as resp:
            if report_status:
                report_status(resp.status)
            if resp.status != 200:
                logger.debug(f"adsb.lol routeset {base_url} returned {resp.status}")
                return None
            data = await resp.json()
    except Exception as exc:
        logger.debug(f"adsb.lol routeset fetch failed for {base_url}: {exc}")
        return None

    results: dict[str, dict | None] = {}
    for entry in data or []:
        if not entry:
            # adsb.lol's routeset response can include a bare `null` entry alongside
            # real ones (confirmed live: AttributeError: 'NoneType' object has no
            # attribute 'get' crashing route enrichment) -- distinct from a callsign
            # simply being absent from the response (this function's own docstring),
            # so skip it rather than raise; the caller's filter_stale retries whatever
            # callsign it represented on a later cycle.
            continue
        callsign = entry.get("callsign")
        if not callsign:
            continue
        airports = entry.get("_airports") or []
        if entry.get("airport_codes") == "unknown" or not airports:
            results[callsign] = None
            continue
        results[callsign] = {
            "stops": [
                {"icao": a.get("icao"), "iata": a.get("iata"), "name": a.get("name")}
                for a in airports
            ],
            "plausible": entry.get("plausible"),
        }
    return results


# --- Global cache-warming sweep (issue #215): GlobalSampleScheduler is what
# AircraftCollector actually uses to decide what to sample each tick -- see that
# class's docstring for how it generalizes the region-keyed due/longest-waiting-first
# shape the removed RegionManager used to implement. ---

# The fine grid shares GRID_DEG (the viewport hot-cell resolution) so an active viewer's
# hot cell lines up exactly with what the frontend itself considers "the area in view."
FINE_GRID_DEG = GRID_DEG

# The background sweep's own, much coarser grid -- required arithmetically, not just for
# convenience: a 30-minute starvation floor (STARVATION_FLOOR_S) at a 6/minute request
# budget can cover at most 180 cells globally (30 * 6). GRID_DEG's own 2,592 cells
# (72 x 36) would need ~14x that budget, or a ~7-hour floor, to keep the same guarantee --
# so the background tier tiles the globe far more coarsely instead. 30deg -> 12 x 6 = 72
# cells, comfortably under budget with headroom for hot-cell traffic interleaved.
COARSE_GRID_DEG = 30.0

HOT_CADENCE_S = 10.0
BACKGROUND_CADENCE_S = 60.0
STARVATION_FLOOR_S = 1800.0

# A background cell needs this many consecutive empty results before its effective
# cadence starts being stretched out (see GlobalSampleScheduler._effective_cadence) --
# below this it's still treated as "unknown", not "reliably empty".
EMPTY_STREAK_THRESHOLD = 3
# Cap on how far a persistently-empty cell's effective cadence can be stretched, so it's
# deprioritized, never fully starved outright (STARVATION_FLOOR_S still forces a
# recheck regardless).
EMPTY_STREAK_MAX_PENALTY = 10.0

# Cap on how many fine-grid cells a single viewport can claim as "hot", so an extremely
# zoomed-out viewport can't blow the request budget by claiming hundreds of cells at
# HOT_CADENCE_S. The cells actually kept are always the ones nearest the viewport
# center -- same nearest-first-under-a-cap shape the old (removed) RegionManager-era
# viewport_to_region_keys used for its gentle tier.
MAX_HOT_CELLS_PER_VIEWPORT = 12


class GlobalSampleScheduler:
    """Pure, now-driven priority queue for AircraftCollector's cache-warming sweep
    (issue #215). Generalizes a due/longest-waiting-first scheduling shape (this
    module's own predecessor, the now-removed RegionManager, used the same idea for
    "regions a WebSocket viewport subscribed to") to the whole globe: a fine grid
    (FINE_GRID_DEG) covers whichever cells currently have an active viewer (per
    set_interest()), sampled at HOT_CADENCE_S; a fixed coarse grid (COARSE_GRID_DEG)
    covers everywhere else, sampled at BACKGROUND_CADENCE_S but adaptively slowed down
    for cells that keep coming back empty. A hard STARVATION_FLOOR_S ceiling protects
    the background grid so no part of the globe goes unsampled indefinitely -- but only
    while nobody is watching (see below); it does not re-admit background cells once a
    viewport has suspended them.

    FlightRadar is a layer someone opens for a while, not something continuously
    watched -- so whenever at least one viewport is active (self._hot_cells non-empty),
    the background sweep is suspended entirely, starvation floor included, and every
    tick's request budget goes to hot cells, not split with the rest of the globe.
    (The floor is scoped to whichever pool is currently eligible, not the full
    hot+background set -- otherwise suspending background is exactly what drives its
    cells past the floor, which would then perpetually re-admit them and undo the
    suspension after ~30 minutes of continuous viewing.) Background sampling only runs
    during idle stretches with no active viewer at all, functioning as cache-warming
    for whenever a viewport next opens (see next_cell()) rather than a
    continuous parallel sweep.

    Not asyncio-aware itself: owns no tasks, does no I/O, and doesn't read viewer
    interest from the database itself -- the caller (AircraftCollector) reads
    AircraftAdapter.get_active_interest() and hands the result to set_interest() each
    cycle. Takes an explicit `now` on every call, a tick-driven-state-machine shape
    (mirroring CollectorBase.is_stale()), so it's testable with controlled timestamps."""

    def __init__(
        self,
        *,
        fine_grid_deg: float = FINE_GRID_DEG,
        coarse_grid_deg: float = COARSE_GRID_DEG,
        hot_cadence_s: float = HOT_CADENCE_S,
        background_cadence_s: float = BACKGROUND_CADENCE_S,
        starvation_floor_s: float = STARVATION_FLOOR_S,
    ):
        self._fine_grid_deg = fine_grid_deg
        self._coarse_grid_deg = coarse_grid_deg
        self._hot_cadence_s = hot_cadence_s
        self._background_cadence_s = background_cadence_s
        self._starvation_floor_s = starvation_floor_s

        # cell key: (grid_deg, ix, iy). Absent from _last_sampled_at => never sampled.
        self._last_sampled_at: dict[tuple, float] = {}
        self._empty_streak: dict[tuple, int] = {}
        # Fine cells backed by at least one active interest viewport as of the most
        # recent set_interest() call -- recomputed fresh every tick, never accumulated.
        self._hot_cells: set[tuple] = set()

    def set_interest(self, viewports: list[tuple[float, float, float, float]]) -> None:
        """Recomputes which fine-grid cells are 'hot' this tick, from the caller's
        fresh read of currently-active viewer interest (west, south, east, north).
        Every fine cell the viewport actually touches becomes hot (capped at
        MAX_HOT_CELLS_PER_VIEWPORT, nearest-to-center first) -- not just the cell at
        its center: "the hotspot" means the whole visible area, with the coarse
        background sweep picking up just outside it, matching this feature's original
        design intent (issue #215).

        Callers are expected to have already filtered out stale/expired interest rows
        (see AircraftAdapter.get_active_interest's max_age_s) -- this method doesn't
        read a clock itself, it just takes whatever's handed to it. Doesn't handle a
        viewport crossing the antimeridian (west > east) -- a known simplification for
        v1, inherited from the pre-issue-#215 viewport_to_region_keys this replaces."""
        hot = set()
        for viewport in viewports:
            hot.update(self._cells_for_viewport(viewport))
        self._hot_cells = hot

    def _cells_for_viewport(self, viewport: tuple[float, float, float, float]) -> list[tuple]:
        """Every fine-grid cell a viewport bbox touches, nearest-to-center first and
        capped at MAX_HOT_CELLS_PER_VIEWPORT -- shared by set_interest() (which only
        needs the resulting set) and hotspot_progress() (which needs this exact same
        per-viewport list to report "N of M cells queried" for just this viewport)."""
        west, south, east, north = viewport
        center_lon, center_lat = (west + east) / 2.0, (south + north) / 2.0
        center = _cell(center_lon, center_lat, self._fine_grid_deg)

        lon_lo, lon_hi = _cell(west, 0.0, self._fine_grid_deg)[0], _cell(east, 0.0, self._fine_grid_deg)[0]
        lat_lo, lat_hi = _cell(0.0, south, self._fine_grid_deg)[1], _cell(0.0, north, self._fine_grid_deg)[1]

        candidates = [
            (lx, ly)
            for lx in range(lon_lo, lon_hi + 1)
            for ly in range(lat_lo, lat_hi + 1)
        ]
        candidates.sort(key=lambda c: (c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2)
        return [
            (self._fine_grid_deg, ix, iy)
            for ix, iy in candidates[:MAX_HOT_CELLS_PER_VIEWPORT]
        ]

    def hotspot_progress(self, viewports: list[tuple[float, float, float, float]]) -> dict:
        """{"queried": n, "total": m} across every fine-grid cell the given viewports
        touch (deduplicated -- two overlapping viewports don't double-count a shared
        cell), "queried" meaning ever sampled at all (not just since becoming hot --
        a cell the background sweep already warmed before anyone looked at it is
        genuinely already populated, not a bug). Callers should pass the SAME
        viewports list just given to set_interest() -- this doesn't read
        self._hot_cells directly since that's a flat union with no per-call viewport
        boundary to report progress against. {"queried": 0, "total": 0} when
        `viewports` is empty (no active viewer to report progress for)."""
        cells: set[tuple] = set()
        for viewport in viewports:
            cells.update(self._cells_for_viewport(viewport))
        total = len(cells)
        queried = sum(1 for c in cells if self._last_sampled_at.get(c) is not None)
        return {"queried": queried, "total": total}

    def global_coverage(self, *, now: float) -> dict:
        """{"fresh": n, "total": m} across the FIXED coarse-grid tiling that covers the
        WHOLE globe (_all_coarse_cells(), independent of any viewer's viewport) --
        "fresh" meaning sampled within the starvation floor window (never overdue for
        its guaranteed recheck), not merely "ever sampled" the way hotspot_progress()
        counts a cell as queried. This answers "how much of the globe currently has
        up-to-date data", the metric AircraftCollector.data_status() surfaces on the
        Data Status Collectors panel -- see that method for why this replaces a plain
        liveness heartbeat there. A never-sampled cell's _elapsed() is +inf, so it
        never counts as fresh."""
        cells = self._all_coarse_cells()
        total = len(cells)
        fresh = sum(1 for c in cells if self._elapsed(c, now=now) < self._starvation_floor_s)
        return {"fresh": fresh, "total": total}

    def _all_coarse_cells(self) -> list[tuple]:
        n_lon = int(360 / self._coarse_grid_deg)
        n_lat = int(180 / self._coarse_grid_deg)
        lon0 = math.floor(-180.0 / self._coarse_grid_deg)
        lat0 = math.floor(-90.0 / self._coarse_grid_deg)
        return [
            (self._coarse_grid_deg, lon0 + i, lat0 + j)
            for i in range(n_lon)
            for j in range(n_lat)
        ]

    def _elapsed(self, cell: tuple, *, now: float) -> float:
        last = self._last_sampled_at.get(cell)
        return float("inf") if last is None else now - last

    def _effective_cadence(self, cell: tuple) -> float:
        """Hot cells are never adaptively deprioritized -- an active viewer's own cell
        should always use hot_cadence_s. Background cells that keep coming back empty
        get a slower effective cadence (capped at EMPTY_STREAK_MAX_PENALTY x), freeing
        budget for cells with actual traffic; the starvation floor still eventually
        forces a recheck regardless of streak."""
        if cell in self._hot_cells:
            return self._hot_cadence_s
        streak = self._empty_streak.get(cell, 0)
        if streak < EMPTY_STREAK_THRESHOLD:
            return self._background_cadence_s
        penalty = min(EMPTY_STREAK_MAX_PENALTY, 1 + (streak - EMPTY_STREAK_THRESHOLD + 1))
        return self._background_cadence_s * penalty

    def next_cell(self, *, now: float) -> tuple | None:
        """The next cell to sample this tick. Whenever at least one viewport is active
        (self._hot_cells non-empty), background cells are excluded from consideration
        entirely -- INCLUDING the starvation floor below -- so the tick either serves a
        due hot cell or, if none is due yet (mid-cadence), goes idle and returns None
        rather than spending that budget on the background sweep. Only once no
        viewport is active at all does the coarse background grid (and its floor
        protection) become eligible again.

        The starvation floor must be scoped to the SAME pool as the cadence check
        below, not the full hot+background candidate set: suspending background
        sampling is exactly what drives its cells past starvation_floor_s in the first
        place (every one of them goes untouched for as long as a viewport stays
        active), so an unscoped floor check would perpetually re-admit the very cells
        this method just excluded -- the hot cell would then never win once a viewing
        session ran past the floor, silently undoing viewport suspension after ~30
        minutes. This was caught live: aircraft in an actively-watched viewport went
        completely stale after roughly starvation_floor_s of continuous viewing.

        Before the hot/background split (this method's previous form), a flat
        oldest-first tie-break across hot+background together degraded into a
        round-robin across the WHOLE combined pool once both tiers were simultaneously
        oversubscribed (the common case) -- an active viewport's own cell then only
        got served once per full cycle (minutes), not at hot_cadence_s. Full
        suspension (rather than merely de-prioritizing background) goes further:
        FlightRadar is watched in bursts, not continuously, so a live viewer should
        get the whole request budget for as long as they're looking, and background
        cache-warming only needs to run in between viewing sessions to keep the globe
        reasonably warm for next time."""
        candidates = set(self._hot_cells) | set(self._all_coarse_cells())
        if not candidates:
            return None

        pool = self._hot_cells if self._hot_cells else candidates

        floored = [c for c in pool if self._elapsed(c, now=now) >= self._starvation_floor_s]
        if floored:
            return min(floored, key=lambda c: self._last_sampled_at.get(c, float("-inf")))

        due = [c for c in pool if self._elapsed(c, now=now) >= self._effective_cadence(c)]
        if not due:
            return None
        return min(due, key=lambda c: self._last_sampled_at.get(c, float("-inf")))

    def record_result(self, cell: tuple, records: list[dict] | None, *, now: float) -> None:
        """records=None (failed fetch) still advances last_sampled_at (so the cell
        backs off at its normal cadence rather than being retried immediately) but
        does NOT count toward the empty streak -- a rejected request isn't evidence a
        cell has no traffic, just that the request failed."""
        self._last_sampled_at[cell] = now
        if records is None:
            return
        if records:
            self._empty_streak[cell] = 0
        else:
            self._empty_streak[cell] = self._empty_streak.get(cell, 0) + 1
