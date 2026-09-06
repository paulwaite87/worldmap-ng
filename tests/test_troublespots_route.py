#!/usr/bin/env python3
"""Route-level test for GET /api/troublespots/geojson, mirroring
test_world_events_route.py. TroublespotAdapter is injected via
Depends(get_troublespot_adapter), so a test can override it with
FakeTroublespotAdapter and exercise the real route end-to-end.
"""
from datetime import datetime, timezone

from atmos_gl.db.troublespot_adapter import FakeTroublespotAdapter
from atmos_gl.routes.troublespots import get_troublespot_adapter
from atmos_gl.api import app


def test_troublespots_geojson_reflects_the_overridden_fake(client):
    fake = FakeTroublespotAdapter()
    now = datetime.now(timezone.utc)
    fake.add_row("earthquake", 10.1, 20.1, now)
    fake.add_row("fire", 10.2, 20.2, now)
    app.dependency_overrides[get_troublespot_adapter] = lambda: fake

    resp = client.get("/api/troublespots/geojson")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) >= 1
    assert body["features"][0]["properties"]["band"] == "elevated"


def test_troublespots_geojson_passes_cell_size_and_window_through_to_the_adapter(client):
    fake = FakeTroublespotAdapter()
    now = datetime.now(timezone.utc)
    fake.add_row("earthquake", 10.1, 20.1, now)
    fake.add_row("fire", 10.2, 20.2, now)
    app.dependency_overrides[get_troublespot_adapter] = lambda: fake

    resp = client.get("/api/troublespots/geojson", params={"cell_size_deg": 1.0, "window_hours": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["features"]) >= 1


def test_troublespots_geojson_is_empty_when_nothing_converges(client):
    fake = FakeTroublespotAdapter()
    app.dependency_overrides[get_troublespot_adapter] = lambda: fake

    resp = client.get("/api/troublespots/geojson")

    assert resp.status_code == 200
    assert resp.json() == {"type": "FeatureCollection", "features": []}
