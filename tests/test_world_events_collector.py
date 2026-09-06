#!/usr/bin/env python3
"""Tests for WorldEventsCollector -- GDELT Event Database 2.0 ingestion, filtered to a
curated CAMEO code allowlist (see collectors/world_events.py's module docstring).
Covers: category classification (including correct exclusions), the min_mentions
noise floor, has_new_data()'s freshness check, and the coverage-based backfill gap-fill
(not a one-shot empty-table gate -- see that method's docstring).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import requests

from atmos_gl.collectors.world_events import WorldEventsCollector, _classify, _parse_export_rows

_BASE_URL = "http://data.gdeltproject.org/gdeltv2"


def make_collector(settings=None):
    c = WorldEventsCollector.__new__(WorldEventsCollector)
    c.settings = settings if settings is not None else {}
    c.world_event_adapter = MagicMock()
    c._etag_cache = {}
    c.datasource_url = MagicMock(return_value=_BASE_URL)
    return c


def _row(
    event_code="183", actor1="", actor2="", lat="10.0", lon="20.0",
    num_mentions="20", num_sources="2", goldstein="1.0", avg_tone="0.5",
    date_added="20260821120000", source_url="http://example.com/a",
    global_event_id="123", action_geo_fullname="Somewhere",
    # Non-blank by default so tests unrelated to actor-vetting keep classifying as
    # before -- see test_classify_conflict_categories_require_a_state_or_group_actor
    # for the case that deliberately blanks these out.
    actor1_country="USA", actor2_country="", actor1_type1="", actor2_type1="",
):
    fields = [""] * 61
    fields[0] = global_event_id
    fields[6] = actor1
    fields[7] = actor1_country
    fields[12] = actor1_type1
    fields[16] = actor2
    fields[17] = actor2_country
    fields[22] = actor2_type1
    fields[26] = event_code
    fields[30] = goldstein
    fields[31] = num_mentions
    fields[32] = num_sources
    fields[34] = avg_tone
    fields[52] = action_geo_fullname
    fields[56] = lat
    fields[57] = lon
    fields[59] = date_added
    fields[60] = source_url
    return "\t".join(fields)


class _FakeResponse:
    def __init__(self, text="", lines=None, status_code=200):
        self.text = text
        self._lines = lines or []
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


# ---- _classify ---------------------------------------------------------------------

def test_classify_explosion_codes():
    assert _classify("183", None, None, True) == "explosion"
    assert _classify("1831", None, None, True) == "explosion"


def test_classify_warfare_codes():
    assert _classify("193", None, None, True) == "warfare"


def test_classify_targeted_violence_codes():
    assert _classify("202", None, None, True) == "targeted_violence"
    assert _classify("2041", None, None, True) == "targeted_violence"


def test_classify_conflict_categories_require_a_state_or_group_actor():
    """The false-positive fix: GDELT's NLP coder can match figurative "battle"/
    "fight"/"war" language (e.g. a film-casting article about "the battle to become
    the next James Bond") into these CAMEO bands. A real conflict's actors resolve to
    a country/military/group entity in CAMEO's dictionaries; an unresolvable proper
    noun doesn't, so has_state_actor=False drops it instead of miscategorizing it."""
    assert _classify("183", "Some Actor", None, False) is None
    assert _classify("193", "Some Actor", None, False) is None
    assert _classify("202", "Some Actor", None, False) is None


def test_classify_diplomacy_requires_both_root_code_and_org_match():
    assert _classify("046", "NATO", "United Kingdom") == "diplomacy"
    # Root-04 event between two ordinary actors, no curated org named -- too generic
    # to count as a World Event (see module docstring).
    assert _classify("046", "France", "Germany") is None


def test_classify_diplomacy_org_match_is_case_insensitive():
    assert _classify("042", "nato secretary general", None) == "diplomacy"


def test_classify_unmatched_code_returns_none():
    assert _classify("010", "Some Actor", None) is None


# ---- _parse_export_rows -------------------------------------------------------------

def test_parse_export_rows_extracts_classified_rows_with_expected_fields():
    csv_text = _row(
        event_code="183", global_event_id="999", lat="51.5", lon="-0.1",
        num_mentions="15", num_sources="3", goldstein="-5.0", avg_tone="-2.3",
        source_url="http://news.example/a", action_geo_fullname="London, UK",
        date_added="20260821123000",
    )
    rows = _parse_export_rows(csv_text)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "999"
    assert row["category"] == "explosion"
    assert row["event_code"] == "183"
    assert row["lat"] == 51.5 and row["lon"] == -0.1
    assert row["num_mentions"] == 15
    assert row["num_sources"] == 3
    assert row["goldstein_scale"] == -5.0
    assert row["avg_tone"] == -2.3
    assert row["source_url"] == "http://news.example/a"
    assert row["action_geo_full_name"] == "London, UK"
    assert row["event_date"] == datetime(2026, 8, 21, 12, 30, tzinfo=timezone.utc).isoformat()


def test_parse_export_rows_excludes_unclassified_and_missing_coords():
    csv_text = "\n".join([
        _row(event_code="010"),           # not in any category allowlist
        _row(event_code="183", lat="", lon=""),  # classified but no coordinates
        _row(event_code="183"),           # valid
    ])
    rows = _parse_export_rows(csv_text)
    assert len(rows) == 1


def test_parse_export_rows_excludes_conflict_code_with_unresolved_actor():
    """Regression test for the GDELT false-positive fix: a warfare-coded event whose
    actors don't resolve to any CAMEO country/military/group entity (Actor*CountryCode
    and Type1-3Code all blank) is dropped, not upserted as "Conflict or War"."""
    csv_text = _row(event_code="193", actor1_country="", actor2_country="")
    rows = _parse_export_rows(csv_text)
    assert rows == []


def test_parse_export_rows_skips_malformed_lines_without_raising():
    csv_text = "\n".join([
        _row(event_code="183", num_mentions="not-a-number"),
        _row(event_code="183"),  # valid
        "too\tfew\tcolumns",
    ])
    rows = _parse_export_rows(csv_text)
    assert len(rows) == 1


def test_parse_export_rows_does_not_apply_a_mentions_floor():
    """min_mentions is applied by the caller (collect()/backfill), not baked into
    parsing -- a row with zero mentions still comes back classified."""
    csv_text = _row(event_code="183", num_mentions="0")
    rows = _parse_export_rows(csv_text)
    assert len(rows) == 1
    assert rows[0]["num_mentions"] == 0


# ---- has_new_data --------------------------------------------------------------

def test_has_new_data_true_when_export_filename_changed():
    c = make_collector()
    lastupdate_text = "123 abc http://data.gdeltproject.org/gdeltv2/20260821120000.export.CSV.zip"
    with patch(
        "atmos_gl.collectors.world_events.requests.get",
        return_value=_FakeResponse(text=lastupdate_text),
    ):
        assert c.has_new_data() is True


def test_has_new_data_false_when_export_filename_unchanged():
    c = make_collector()
    url = "http://data.gdeltproject.org/gdeltv2/20260821120000.export.CSV.zip"
    c._etag_cache[f"{_BASE_URL}/lastupdate.txt"] = url
    lastupdate_text = f"123 abc {url}"
    with patch(
        "atmos_gl.collectors.world_events.requests.get",
        return_value=_FakeResponse(text=lastupdate_text),
    ):
        assert c.has_new_data() is False


def test_has_new_data_defaults_true_on_fetch_failure():
    c = make_collector()
    with patch(
        "atmos_gl.collectors.world_events.requests.get",
        side_effect=requests.ConnectionError("down"),
    ):
        assert c.has_new_data() is True


# ---- collect (backfill disabled via full coverage, to isolate the normal-cycle path) --

def test_collect_upserts_rows_at_or_above_the_mentions_floor():
    c = make_collector(settings={"min_mentions": 10, "backfill_days": 3})
    c.world_event_adapter.oldest_event_date.return_value = datetime.now(timezone.utc)
    lastupdate_text = f"123 abc {_BASE_URL}/20260821120000.export.CSV.zip"
    csv_text = "\n".join([
        _row(event_code="183", global_event_id="1", num_mentions="20"),
        _row(event_code="193", global_event_id="2", num_mentions="5"),  # below floor
    ])

    with patch("atmos_gl.collectors.world_events.requests.get", return_value=_FakeResponse(text=lastupdate_text)), \
         patch("atmos_gl.collectors.world_events._fetch_export_csv", return_value=csv_text):
        c.collect()

    (upserted_rows,), _ = c.world_event_adapter.upsert_events.call_args
    assert [r["id"] for r in upserted_rows] == ["1"]
    assert c._etag_cache[f"{_BASE_URL}/lastupdate.txt"] == f"{_BASE_URL}/20260821120000.export.CSV.zip"


def test_collect_skips_when_lastupdate_has_no_export_entry():
    c = make_collector()
    c.world_event_adapter.oldest_event_date.return_value = datetime.now(timezone.utc)
    with patch(
        "atmos_gl.collectors.world_events.requests.get",
        return_value=_FakeResponse(text="no export file here"),
    ):
        c.collect()
    c.world_event_adapter.upsert_events.assert_not_called()


# ---- backfill gap-fill ----------------------------------------------------------

def test_backfill_skipped_when_coverage_already_reaches_backfill_days():
    c = make_collector(settings={"backfill_days": 3})
    c.world_event_adapter.oldest_event_date.return_value = (
        datetime.now(timezone.utc) - timedelta(days=5)
    )
    with patch("atmos_gl.collectors.world_events.requests.get") as mock_get:
        c._backfill_gap(backfill_days=3, min_mentions=10)
    mock_get.assert_not_called()


def test_backfill_fetches_only_the_missing_window_on_an_empty_table():
    c = make_collector(settings={"backfill_days": 1})
    c.world_event_adapter.oldest_event_date.return_value = None
    now = datetime.now(timezone.utc)

    def ts(delta_hours):
        return (now - timedelta(hours=delta_hours)).strftime("%Y%m%d%H%M%S")

    master_lines = [
        # Well outside the 1-day backfill window -- must be skipped.
        f"1 x {_BASE_URL}/{ts(24 * 30)}.export.CSV.zip",
        # Inside the window.
        f"1 x {_BASE_URL}/{ts(20)}.export.CSV.zip",
        f"1 x {_BASE_URL}/{ts(10)}.export.CSV.zip",
        # Not an export file -- must be ignored.
        f"1 x {_BASE_URL}/{ts(5)}.mentions.CSV.zip",
    ]
    master_response = _FakeResponse(lines=master_lines)
    csv_text = _row(event_code="183", global_event_id="42", num_mentions="20")

    def fake_get(url, **kwargs):
        if url.endswith("masterfilelist.txt"):
            return master_response
        raise AssertionError(f"unexpected fetch: {url}")

    with patch("atmos_gl.collectors.world_events.requests.get", side_effect=fake_get), \
         patch("atmos_gl.collectors.world_events._fetch_export_csv", return_value=csv_text) as mock_fetch_csv:
        c._backfill_gap(backfill_days=1, min_mentions=10)

    assert mock_fetch_csv.call_count == 2
    fetched_urls = {call.args[0] for call in mock_fetch_csv.call_args_list}
    assert all(".export.CSV.zip" in u for u in fetched_urls)
    assert c.world_event_adapter.upsert_events.call_count == 2


def test_backfill_one_file_failing_does_not_block_the_rest():
    c = make_collector(settings={"backfill_days": 1})
    c.world_event_adapter.oldest_event_date.return_value = None
    now = datetime.now(timezone.utc)

    def ts(delta_hours):
        return (now - timedelta(hours=delta_hours)).strftime("%Y%m%d%H%M%S")

    master_lines = [
        f"1 x {_BASE_URL}/{ts(20)}.export.CSV.zip",
        f"1 x {_BASE_URL}/{ts(10)}.export.CSV.zip",
    ]

    def fake_get(url, **kwargs):
        if url.endswith("masterfilelist.txt"):
            return _FakeResponse(lines=master_lines)
        raise AssertionError(f"unexpected fetch: {url}")

    def fake_fetch_csv(url):
        if ts(20) in url:
            raise requests.ConnectionError("mid-backfill outage")
        return _row(event_code="183", global_event_id="ok", num_mentions="20")

    with patch("atmos_gl.collectors.world_events.requests.get", side_effect=fake_get), \
         patch("atmos_gl.collectors.world_events._fetch_export_csv", side_effect=fake_fetch_csv):
        c._backfill_gap(backfill_days=1, min_mentions=10)  # must not raise

    assert c.world_event_adapter.upsert_events.call_count == 1
