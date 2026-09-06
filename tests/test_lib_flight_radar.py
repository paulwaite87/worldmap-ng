#!/usr/bin/env python3
"""Tests for lib/flight_radar.py -- the pure geometry helpers behind Flight Radar's
data acquisition (issue #203/#215). No real network in any of these -- fetch_aircraft_near
is exercised against fake aiohttp-shaped sessions.

RegionManager and viewport_to_region_keys (the WebSocket-era subscription lifecycle and
viewport-to-hot/gentle-key mapping, docs/adr/0009) were removed once AircraftCollector
replaced the WS route as adsb.lol's sole consumer -- see docs/adr/0010 and
tests/test_global_sample_scheduler.py for their replacement, GlobalSampleScheduler."""
import pytest

from atmos_gl.lib.flight_radar import circle_for_region_key, fetch_aircraft_near, fetch_routes


class _FakeResponse:
    """Mimics aiohttp's response context manager: `session.get(...)` returns this
    directly (not a coroutine), and `async with ... as resp` drives it."""

    def __init__(self, status, json_body=None):
        self.status = status
        self._json_body = json_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def json(self):
        return self._json_body


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.last_url = None
        self.last_json = None

    def get(self, url, timeout=None):
        self.last_url = url
        return self._response

    def post(self, url, json=None, timeout=None):
        self.last_url = url
        self.last_json = json
        return self._response


class _RaisingSession:
    def get(self, url, timeout=None):
        raise RuntimeError("connection failed")

    def post(self, url, json=None, timeout=None):
        raise RuntimeError("connection failed")


def test_circle_for_region_key_centers_on_the_cell_center():
    lat, lon, _radius = circle_for_region_key((0, 0), grid_deg=5.0)
    assert lon == pytest.approx(2.5)
    assert lat == pytest.approx(2.5)


def test_circle_for_region_key_handles_negative_cells():
    lat, lon, _radius = circle_for_region_key((-1, -1), grid_deg=5.0)
    assert lon == pytest.approx(-2.5)
    assert lat == pytest.approx(-2.5)


def test_circle_for_region_key_uses_the_configured_radius():
    _lat, _lon, radius = circle_for_region_key((0, 0), grid_deg=5.0, radius_nm=123.0)
    assert radius == 123.0


# ---- fetch_aircraft_near: success vs. confirmed-empty vs. failed -----------------
# adsb.lol's free tier 429s far more readily than a naively-assumed cadence would
# expect -- a rejected/failed request must come back as None, distinct from a real []
# (a request that succeeded and genuinely found no aircraft in range).

@pytest.mark.asyncio
async def test_fetch_aircraft_near_returns_the_ac_list_on_success():
    session = _FakeSession(_FakeResponse(200, {"ac": [{"hex": "a1"}]}))
    result = await fetch_aircraft_near(session, 0.0, 0.0, 200.0)
    assert result == [{"hex": "a1"}]


@pytest.mark.asyncio
async def test_fetch_aircraft_near_returns_an_empty_list_when_ac_key_is_absent():
    """A 200 with no aircraft in range is a real, confirmed-empty result -- unlike a
    rejected request, it's fine to report this as []."""
    session = _FakeSession(_FakeResponse(200, {}))
    result = await fetch_aircraft_near(session, 0.0, 0.0, 200.0)
    assert result == []


@pytest.mark.asyncio
async def test_fetch_aircraft_near_returns_none_not_empty_on_a_non_200_status():
    session = _FakeSession(_FakeResponse(429))
    result = await fetch_aircraft_near(session, 0.0, 0.0, 200.0)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_aircraft_near_returns_none_on_a_raised_exception():
    result = await fetch_aircraft_near(_RaisingSession(), 0.0, 0.0, 200.0)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_aircraft_near_uses_the_default_base_url_when_not_overridden():
    session = _FakeSession(_FakeResponse(200, {"ac": []}))
    await fetch_aircraft_near(session, 0.0, 0.0, 200.0)
    assert session.last_url.startswith("https://api.adsb.lol/v2/")


@pytest.mark.asyncio
async def test_fetch_aircraft_near_honors_a_configured_base_url():
    """AircraftCollector passes flightradar_collector's configured
    data_collector.datasources.flightradar value here rather than always using the
    hardcoded ADSB_LOL_BASE default -- the same maintainable-datasources-list
    convention every other collector follows."""
    session = _FakeSession(_FakeResponse(200, {"ac": []}))
    await fetch_aircraft_near(session, 0.0, 0.0, 200.0, base_url="https://my-mirror.example/v2")
    assert session.last_url.startswith("https://my-mirror.example/v2/")


# ---- report_status: the health-reporting side-channel (issue #215's Data Status
# health icons) -- independent of the records None/[]/list contract above.

@pytest.mark.asyncio
async def test_fetch_aircraft_near_reports_status_on_success():
    session = _FakeSession(_FakeResponse(200, {"ac": []}))
    reported = []
    await fetch_aircraft_near(session, 0.0, 0.0, 200.0, report_status=reported.append)
    assert reported == [200]


@pytest.mark.asyncio
async def test_fetch_aircraft_near_reports_status_on_rate_limit():
    session = _FakeSession(_FakeResponse(429))
    reported = []
    await fetch_aircraft_near(session, 0.0, 0.0, 200.0, report_status=reported.append)
    assert reported == [429]


@pytest.mark.asyncio
async def test_fetch_aircraft_near_never_reports_status_on_a_raised_exception():
    """A timeout/connection error never gets a response at all -- there's no status
    code to report, unlike a real rejection (429/5xx)."""
    reported = []
    await fetch_aircraft_near(_RaisingSession(), 0.0, 0.0, 200.0, report_status=reported.append)
    assert reported == []


@pytest.mark.asyncio
async def test_fetch_aircraft_near_works_without_a_report_status_callback():
    """report_status is optional -- omitting it (the default None) must not raise."""
    session = _FakeSession(_FakeResponse(200, {"ac": [{"hex": "a1"}]}))
    result = await fetch_aircraft_near(session, 0.0, 0.0, 200.0)
    assert result == [{"hex": "a1"}]


# ---- fetch_routes: batch callsign -> route lookup (issue #215's route-lookup
# follow-on), verified against adsblol/api's real server source
# (src/adsb_api/utils/api_routes.py) rather than guessed.

def _airport(icao, iata, name="Some Airport"):
    return {"icao": icao, "iata": iata, "name": name}


@pytest.mark.asyncio
async def test_fetch_routes_returns_empty_dict_for_an_empty_batch_without_a_request():
    session = _FakeSession(_FakeResponse(200, []))
    result = await fetch_routes(session, [])
    assert result == {}
    assert session.last_url is None


@pytest.mark.asyncio
async def test_fetch_routes_matches_a_response_entry_by_its_callsign_field():
    """Matched by "callsign", not array position -- adsblol/api's api_routeset echoes
    the callsign back onto every entry it builds."""
    body = [
        {
            "callsign": "ANZ423",
            "airport_codes": "WGN-AKL",
            "_airports": [_airport("NZWN", "WGN"), _airport("NZAA", "AKL")],
            "plausible": True,
        }
    ]
    session = _FakeSession(_FakeResponse(200, body))
    result = await fetch_routes(session, [{"callsign": "ANZ423", "lat": -41.3, "lng": 174.8}])
    assert result == {
        "ANZ423": {
            "stops": [
                {"icao": "NZWN", "iata": "WGN", "name": "Some Airport"},
                {"icao": "NZAA", "iata": "AKL", "name": "Some Airport"},
            ],
            "plausible": True,
        }
    }


@pytest.mark.asyncio
async def test_fetch_routes_preserves_an_intermediate_stop_in_order():
    """A technical-stop route (3 _airports entries) keeps the WHOLE ordered list, not
    just origin/destination -- this feature's Q8/Q9 design decision."""
    body = [
        {
            "callsign": "ANZ423",
            "airport_codes": "WGN-CHC-AKL",
            "_airports": [_airport("NZWN", "WGN"), _airport("NZCH", "CHC"), _airport("NZAA", "AKL")],
            "plausible": True,
        }
    ]
    session = _FakeSession(_FakeResponse(200, body))
    result = await fetch_routes(session, [{"callsign": "ANZ423", "lat": 0.0, "lng": 0.0}])
    assert [s["iata"] for s in result["ANZ423"]["stops"]] == ["WGN", "CHC", "AKL"]


@pytest.mark.asyncio
async def test_fetch_routes_confirmed_no_match_is_none_not_missing():
    """adsblol/api's own no-match sentinel: "airport_codes": "unknown", "_airports": []."""
    body = [{"callsign": "N12345", "airport_codes": "unknown", "_airports": []}]
    session = _FakeSession(_FakeResponse(200, body))
    result = await fetch_routes(session, [{"callsign": "N12345", "lat": 0.0, "lng": 0.0}])
    assert result == {"N12345": None}


@pytest.mark.asyncio
async def test_fetch_routes_skips_a_bare_null_entry_in_the_response_list():
    """Confirmed live: adsb.lol's routeset response can include a bare `null` entry
    alongside real ones -- entry.get(...) on it raised AttributeError and crashed
    route enrichment entirely (not just that one callsign). A null entry must be
    skipped, not misread as "every callsign in this batch has no route" and not
    allowed to crash the whole call."""
    body = [None, {"callsign": "N12345", "_airports": [{"icao": "NZAA"}, {"icao": "NZQN"}]}]
    session = _FakeSession(_FakeResponse(200, body))
    result = await fetch_routes(
        session,
        [
            {"callsign": "UNKNOWN1", "lat": 0.0, "lng": 0.0},
            {"callsign": "N12345", "lat": 0.0, "lng": 0.0},
        ],
    )
    assert result == {
        "N12345": {
            "stops": [
                {"icao": "NZAA", "iata": None, "name": None},
                {"icao": "NZQN", "iata": None, "name": None},
            ],
            "plausible": None,
        }
    }


@pytest.mark.asyncio
async def test_fetch_routes_returns_none_for_the_whole_batch_on_a_non_200_status():
    """Distinct from a per-callsign no-match: a rejected/failed BATCH request must
    never be misread as "every callsign in it has no route"."""
    session = _FakeSession(_FakeResponse(429))
    result = await fetch_routes(session, [{"callsign": "ANZ423", "lat": 0.0, "lng": 0.0}])
    assert result is None


@pytest.mark.asyncio
async def test_fetch_routes_returns_none_on_a_raised_exception():
    result = await fetch_routes(_RaisingSession(), [{"callsign": "ANZ423", "lat": 0.0, "lng": 0.0}])
    assert result is None


@pytest.mark.asyncio
async def test_fetch_routes_sends_the_real_position_not_a_placeholder():
    """Real lat/lng (not 0/0) is what lets adsb.lol compute its own "plausible"
    great-circle sanity check server-side."""
    session = _FakeSession(_FakeResponse(200, []))
    await fetch_routes(session, [{"callsign": "ANZ423", "lat": -41.3, "lng": 174.8}])
    assert session.last_json == {"planes": [{"callsign": "ANZ423", "lat": -41.3, "lng": 174.8}]}


@pytest.mark.asyncio
async def test_fetch_routes_batches_multiple_callsigns_in_one_request():
    body = [
        {"callsign": "ANZ423", "airport_codes": "unknown", "_airports": []},
        {
            "callsign": "QFA144",
            "airport_codes": "AKL-SYD",
            "_airports": [_airport("NZAA", "AKL"), _airport("YSSY", "SYD")],
            "plausible": False,
        },
    ]
    session = _FakeSession(_FakeResponse(200, body))
    result = await fetch_routes(
        session,
        [
            {"callsign": "ANZ423", "lat": 0.0, "lng": 0.0},
            {"callsign": "QFA144", "lat": 0.0, "lng": 0.0},
        ],
    )
    assert result["ANZ423"] is None
    assert result["QFA144"]["plausible"] is False
    assert [s["iata"] for s in result["QFA144"]["stops"]] == ["AKL", "SYD"]


@pytest.mark.asyncio
async def test_fetch_routes_uses_the_default_base_url_when_not_overridden():
    """adsb.im, not api.adsb.lol -- see ADSB_LOL_ROUTESET_BASE's comment: the latter's
    routeset endpoint was verified broken (bare 201, empty body) as of 2026-07-26."""
    session = _FakeSession(_FakeResponse(200, []))
    await fetch_routes(session, [{"callsign": "ANZ423", "lat": 0.0, "lng": 0.0}])
    assert session.last_url == "https://adsb.im/api/0/routeset"


@pytest.mark.asyncio
async def test_fetch_routes_honors_a_configured_base_url():
    session = _FakeSession(_FakeResponse(200, []))
    await fetch_routes(
        session, [{"callsign": "ANZ423", "lat": 0.0, "lng": 0.0}],
        base_url="https://my-mirror.example/api/0/routeset",
    )
    assert session.last_url == "https://my-mirror.example/api/0/routeset"


@pytest.mark.asyncio
async def test_fetch_routes_reports_status_on_success():
    session = _FakeSession(_FakeResponse(200, []))
    reported = []
    await fetch_routes(
        session, [{"callsign": "ANZ423", "lat": 0.0, "lng": 0.0}], report_status=reported.append
    )
    assert reported == [200]


@pytest.mark.asyncio
async def test_fetch_routes_never_reports_status_on_a_raised_exception():
    reported = []
    await fetch_routes(
        _RaisingSession(), [{"callsign": "ANZ423", "lat": 0.0, "lng": 0.0}], report_status=reported.append
    )
    assert reported == []
