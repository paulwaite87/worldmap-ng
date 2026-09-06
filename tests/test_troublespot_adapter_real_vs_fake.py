#!/usr/bin/env python3
"""Guard against TroublespotAdapter Real/Fake drift, mirroring
test_world_event_adapter_real_vs_fake.py. Troublespots (issue #366) has no table of
its own -- it's a derived view over Earthquakes/Fires/Volcanic Activity/World Events --
so "real" seeds those four tables directly via SQLAlchemy (bypassing their own
per-domain adapters, since these tests only need bare lat/lon/timestamp rows) while
"fake" seeds FakeTroublespotAdapter's in-memory add_row(). Both then exercise the SAME
compute_troublespot_bands() (tests/test_troublespot_contours.py already covers that
math directly) -- what these tests guard is the binning/breakdown step matching
between the two.
"""
import contextlib
import itertools
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

# real_db is session-scoped (shared across every test in this module) -- a global
# counter keeps every inserted row's id unique across test functions, not just within
# one _seed_real() call, mirroring how the world_event/fire real-vs-fake tests suffix
# ids with the test's own name to avoid the same collision.
_row_id_counter = itertools.count()

from atmos_gl.db.models import Earthquake, Fire, VolcanicActivity, WorldEvent
from atmos_gl.db.troublespot_adapter import FakeTroublespotAdapter, TroublespotAdapter


def _make_adapter(kind, real_db):
    if kind == "real":
        TestSession = sessionmaker(bind=real_db)
        return TroublespotAdapter(), patch(
            "atmos_gl.db.troublespot_adapter.Session", TestSession
        )
    return FakeTroublespotAdapter(), contextlib.nullcontext()


def _seed_real(real_db, rows):
    """rows: list of (source_type, lat, lon, timestamp). Inserts bare rows directly
    into whichever of the four tables source_type maps to -- these tests only need
    lat/lon/the type's own time column populated, not a full realistic row."""
    TestSession = sessionmaker(bind=real_db)
    with TestSession() as session:
        for source_type, lat, lon, ts in rows:
            i = next(_row_id_counter)
            geom = f"SRID=4326;POINT({lon} {lat})"
            if source_type == "earthquake":
                session.add(Earthquake(id=f"ts-quake-{i}", lat=lat, lon=lon, geom=geom, eq_time=ts))
            elif source_type == "fire":
                session.add(Fire(id=f"ts-fire-{i}", lat=lat, lon=lon, geom=geom, acq_time=ts))
            elif source_type == "volcanic_activity":
                session.add(
                    VolcanicActivity(
                        vnum=f"ts-volcano-{i}", lat=lat, lon=lon, geom=geom, last_seen_at=ts
                    )
                )
            elif source_type == "world_event":
                session.add(
                    WorldEvent(
                        id=f"ts-event-{i}", category="conflict", lat=lat, lon=lon,
                        geom=geom, event_date=ts,
                    )
                )
            else:
                raise ValueError(source_type)
        session.commit()


def _seed(kind, adapter, real_db, rows):
    if kind == "real":
        _seed_real(real_db, rows)
    else:
        for source_type, lat, lon, ts in rows:
            adapter.add_row(source_type, lat, lon, ts)


def _feature_near(geojson, lon, lat, tolerance_deg=1.0):
    """The feature whose polygon's centroid is within tolerance_deg of (lon, lat), or
    None -- real_db is session-scoped (shared across every test in this module), so
    asserting on the FULL feature list (order or exact count) is unsafe; each test
    instead looks for its own result near its own well-separated coordinates."""
    for f in geojson["features"]:
        ring = f["geometry"]["coordinates"][0]
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        if abs(cx - lon) <= tolerance_deg and abs(cy - lat) <= tolerance_deg:
            return f
    return None


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_two_converging_types_produce_a_troublespot(kind, real_db):
    adapter, ctx = _make_adapter(kind, real_db)
    now = datetime.now(timezone.utc)

    with ctx:
        _seed(kind, adapter, real_db, [
            ("earthquake", 10.1, 20.1, now),
            ("fire", 10.2, 20.2, now),
        ])
        geojson = json.loads(adapter.get_troublespots_as_geojson(cell_size_deg=2.0, window_hours=48))

    feature = _feature_near(geojson, 20.15, 10.15)
    assert feature is not None
    assert feature["properties"]["band"] == "elevated"


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_a_single_type_alone_never_produces_a_troublespot(kind, real_db):
    adapter, ctx = _make_adapter(kind, real_db)
    now = datetime.now(timezone.utc)

    with ctx:
        _seed(kind, adapter, real_db, [
            ("fire", -40.1, 100.1, now),
            ("fire", -40.2, 100.2, now),
            ("fire", -40.3, 100.3, now),
        ])
        geojson = json.loads(adapter.get_troublespots_as_geojson(cell_size_deg=2.0, window_hours=48))

    assert _feature_near(geojson, 100.2, -40.2) is None


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_rows_outside_the_time_window_are_excluded(kind, real_db):
    adapter, ctx = _make_adapter(kind, real_db)
    now = datetime.now(timezone.utc)
    stale = now - timedelta(hours=200)

    with ctx:
        _seed(kind, adapter, real_db, [
            ("earthquake", 55.1, -70.1, stale),
            ("fire", 55.2, -70.2, stale),
        ])
        geojson = json.loads(adapter.get_troublespots_as_geojson(cell_size_deg=2.0, window_hours=48))

    assert _feature_near(geojson, -70.15, 55.15) is None


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_breakdown_counts_reflect_rows_actually_inside_the_polygon(kind, real_db):
    adapter, ctx = _make_adapter(kind, real_db)
    now = datetime.now(timezone.utc)

    with ctx:
        _seed(kind, adapter, real_db, [
            ("earthquake", 30.1, 60.1, now),
            ("earthquake", 30.3, 60.3, now),
            ("fire", 30.2, 60.2, now),
            ("world_event", 30.4, 60.4, now),
        ])
        geojson = json.loads(adapter.get_troublespots_as_geojson(cell_size_deg=2.0, window_hours=48))

    feature = _feature_near(geojson, 60.2, 30.2)
    assert feature is not None
    props = feature["properties"]
    assert props["earthquake"] == 2
    assert props["fire"] == 1
    assert props["world_event"] == 1
    assert props["volcanic_activity"] == 0
