#!/usr/bin/env python3
"""Route-level test for GET /api/world_events/geojson, mirroring test_fires_route.py.
WorldEventAdapter is injected via Depends(get_world_event_adapter), so a test can
override it with FakeWorldEventAdapter and exercise the real route end-to-end.
"""
from datetime import datetime, timedelta, timezone

from atmos_gl.db.world_event_adapter import FakeWorldEventAdapter
from atmos_gl.routes.world_events import get_world_event_adapter
from atmos_gl.api import app


def _event(event_id, event_date_iso, category="warfare"):
    return {
        "id": event_id, "category": category, "event_code": "193",
        "actor1_name": "Actor A", "actor2_name": "Actor B",
        "action_geo_full_name": "Somewhere", "lat": 10.0, "lon": 20.0,
        "event_date": event_date_iso, "num_mentions": 15, "num_sources": 2,
        "goldstein_scale": -5.0, "avg_tone": -2.0, "source_url": "http://example.com/a",
    }


def test_world_events_geojson_reflects_the_overridden_fake(client):
    fake = FakeWorldEventAdapter()
    now_iso = datetime.now(timezone.utc).isoformat()
    fake.upsert_events([_event("e1", now_iso)])
    app.dependency_overrides[get_world_event_adapter] = lambda: fake

    resp = client.get("/api/world_events/geojson")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 1
    props = body["features"][0]["properties"]
    assert props["id"] == "e1"
    assert props["category"] == "warfare"
    assert props["actor1_name"] == "Actor A"


def test_world_events_geojson_passes_expiry_days_through_to_the_adapter(client):
    fake = FakeWorldEventAdapter()
    now = datetime.now(timezone.utc)
    fake.upsert_events([
        _event("recent", now.isoformat()),
        _event("old", (now - timedelta(days=30)).isoformat()),
    ])
    app.dependency_overrides[get_world_event_adapter] = lambda: fake

    resp = client.get("/api/world_events/geojson", params={"expiry_days": 7})

    ids = {f["properties"]["id"] for f in resp.json()["features"]}
    assert ids == {"recent"}
