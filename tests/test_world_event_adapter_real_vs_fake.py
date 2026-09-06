#!/usr/bin/env python3
"""Guard against WorldEventAdapter Real/Fake drift, mirroring
test_fire_adapter_real_vs_fake.py: FakeWorldEventAdapter hand-reimplements
WorldEventAdapter's on-conflict SQL and expiry filter independently, so if they ever
diverge, nothing else would catch it. Unlike FireAdapter (whose lat/lon/geom is
immutable on conflict), every column here refreshes on a re-ingested GLOBALEVENTID --
GDELT's own fields for a given event don't change after the fact, so there's no
"protect the original coordinate" concern a backfill re-covering an already-collected
window needs to respect.
"""
import contextlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from atmos_gl.db.world_event_adapter import WorldEventAdapter, FakeWorldEventAdapter


def _make_adapter(kind, real_db):
    if kind == "real":
        TestSession = sessionmaker(bind=real_db)
        return WorldEventAdapter(), patch("atmos_gl.db.world_event_adapter.Session", TestSession)
    return FakeWorldEventAdapter(), contextlib.nullcontext()


def _row(adapter, event_id, real_db):
    if isinstance(adapter, FakeWorldEventAdapter):
        e = adapter._events[event_id]
        return {"category": e["category"], "num_mentions": e["num_mentions"], "lat": e["lat"], "lon": e["lon"]}
    with real_db.connect() as conn:
        result = conn.execute(
            text("SELECT category, num_mentions, lat, lon FROM world_events WHERE id = :id"),
            {"id": event_id},
        ).mappings().one()
        return dict(result)


def _event_row(event_id, category, lat, lon, event_date_iso, num_mentions=20, event_code="183"):
    return {
        "id": event_id, "category": category, "event_code": event_code,
        "actor1_name": None, "actor2_name": None, "action_geo_full_name": None,
        "lat": lat, "lon": lon, "event_date": event_date_iso,
        "num_mentions": num_mentions, "num_sources": 1,
        "goldstein_scale": None, "avg_tone": None, "source_url": None,
    }


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_every_column_refreshes_on_conflict(kind, real_db):
    event_id = f"we-update-{kind}"
    adapter, ctx = _make_adapter(kind, real_db)
    now_iso = datetime.now(timezone.utc).isoformat()

    with ctx:
        adapter.upsert_events([_event_row(event_id, "warfare", 10.0, 20.0, now_iso, num_mentions=5)])
        adapter.upsert_events([_event_row(event_id, "explosion", 30.0, 40.0, now_iso, num_mentions=99)])
        row = _row(adapter, event_id, real_db)

    assert row["category"] == "explosion"
    assert row["num_mentions"] == 99
    assert row["lat"] == pytest.approx(30.0)
    assert row["lon"] == pytest.approx(40.0)


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_rows_missing_coordinates_are_never_stored(kind, real_db):
    event_id = f"we-nocoord-{kind}"
    adapter, ctx = _make_adapter(kind, real_db)
    now_iso = datetime.now(timezone.utc).isoformat()
    row = _event_row(event_id, "warfare", None, None, now_iso)

    with ctx:
        adapter.upsert_events([row])
        geojson = json.loads(adapter.get_events_as_geojson(expiry_days=36500))

    ids = {f["properties"]["id"] for f in geojson["features"]}
    assert event_id not in ids


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_get_events_as_geojson_expiry_filter_matches_between_real_and_fake(kind, real_db):
    suffix = kind
    adapter, ctx = _make_adapter(kind, real_db)
    now = datetime.now(timezone.utc)
    recent_iso = now.isoformat()
    old_iso = (now - timedelta(days=30)).isoformat()

    with ctx:
        adapter.upsert_events([
            _event_row(f"recent-{suffix}", "warfare", 10.0, 20.0, recent_iso),
            _event_row(f"old-{suffix}", "warfare", 10.0, 20.0, old_iso),
        ])
        geojson = json.loads(adapter.get_events_as_geojson(expiry_days=7))

    ids = {f["properties"]["id"] for f in geojson["features"]}
    assert f"recent-{suffix}" in ids
    assert f"old-{suffix}" not in ids


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_delete_expired_prunes_only_rows_older_than_expiry_days(kind, real_db):
    suffix = kind
    adapter, ctx = _make_adapter(kind, real_db)
    now = datetime.now(timezone.utc)

    with ctx:
        adapter.upsert_events([
            _event_row(f"keep-{suffix}", "warfare", 10.0, 20.0, now.isoformat()),
            _event_row(f"prune-{suffix}", "warfare", 10.0, 20.0, (now - timedelta(days=10)).isoformat()),
        ])
        deleted = adapter.delete_expired(expiry_days=7)
        geojson = json.loads(adapter.get_events_as_geojson(expiry_days=36500))

    # >=1 rather than an exact count: the real_db fixture's table is shared across
    # this module's tests, so an earlier test's own aged-out row(s) may also get
    # swept up here -- what matters is that THIS test's prune row is gone and its
    # keep row survived, not the total row count deleted.
    assert deleted >= 1
    ids = {f["properties"]["id"] for f in geojson["features"]}
    assert f"keep-{suffix}" in ids
    assert f"prune-{suffix}" not in ids
