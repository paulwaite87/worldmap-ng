import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import cast, func, select, delete
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.types import Text as SqlText

from atmos_gl.db.engine import Session
from atmos_gl.db.geojson import as_feature_collection, EMPTY_FEATURE_COLLECTION
from atmos_gl.db.models import WorldEvent

logger = logging.getLogger(__name__)

# Same chunking rationale as FireAdapter/MarkerAdapter's bulk upserts: a backfill run
# can cover several days of curated-category GDELT events in one collect() cycle, well
# past a single-statement-per-row shape's comfortable size.
_UPSERT_CHUNK_SIZE = 5000


class WorldEventAdapter:
    """Real adapter for world_events (GDELT-sourced conflict/diplomacy point markers,
    see collectors/world_events.py), backed by SQLAlchemy."""

    def upsert_events(self, rows):
        """Bulk-UPSERTs classified GDELT events. Each row is a dict with keys: id
        (GLOBALEVENTID), category, event_code, actor1_name, actor2_name,
        action_geo_full_name, lat, lon, event_date (ISO str), num_mentions,
        num_sources, goldstein_scale, avg_tone, source_url.

        A re-ingested id (a backfill re-covering a window collect() already filled)
        just refreshes the same row -- GDELT's own fields for a given event don't
        change after the fact, so the ON CONFLICT SET list is the full column set."""
        if not rows:
            return
        values = [
            {**r, "geom": f"SRID=4326;POINT({r['lon']} {r['lat']})"}
            for r in rows
            if r.get("lat") is not None and r.get("lon") is not None
        ]
        if not values:
            return
        stmt = pg_insert(WorldEvent)
        stmt = stmt.on_conflict_do_update(
            index_elements=[WorldEvent.id],
            set_={
                "category": stmt.excluded.category,
                "event_code": stmt.excluded.event_code,
                "actor1_name": stmt.excluded.actor1_name,
                "actor2_name": stmt.excluded.actor2_name,
                "action_geo_full_name": stmt.excluded.action_geo_full_name,
                "lat": stmt.excluded.lat,
                "lon": stmt.excluded.lon,
                "geom": stmt.excluded.geom,
                "event_date": stmt.excluded.event_date,
                "num_mentions": stmt.excluded.num_mentions,
                "num_sources": stmt.excluded.num_sources,
                "goldstein_scale": stmt.excluded.goldstein_scale,
                "avg_tone": stmt.excluded.avg_tone,
                "source_url": stmt.excluded.source_url,
            },
        )
        try:
            with Session() as session:
                for i in range(0, len(values), _UPSERT_CHUNK_SIZE):
                    session.execute(stmt, values[i : i + _UPSERT_CHUNK_SIZE])
                session.commit()
        except Exception as e:
            logger.error(f"Error bulk-saving {len(values)} world events: {e}")

    def oldest_event_date(self):
        """The earliest event_date currently stored, or None if the table is empty --
        used by WorldEventsCollector to decide how much of backfill_days' coverage is
        still missing (see that module's docstring)."""
        try:
            with Session() as session:
                return session.scalar(select(func.min(WorldEvent.event_date)))
        except Exception as e:
            logger.error(f"Error reading oldest world event date: {e}")
            return None

    def get_events_as_geojson(self, expiry_days=7):
        """Returns world events as GeoJSON, filtering by age. age_hours (like quakes'
        age_minutes) lets the frontend de-emphasize older events within the window."""
        feature = func.jsonb_build_object(
            "type",
            "Feature",
            "geometry",
            cast(func.ST_AsGeoJSON(WorldEvent.geom), JSONB),
            "properties",
            func.jsonb_build_object(
                "id",
                WorldEvent.id,
                "category",
                WorldEvent.category,
                "event_code",
                WorldEvent.event_code,
                "actor1_name",
                WorldEvent.actor1_name,
                "actor2_name",
                WorldEvent.actor2_name,
                "place",
                WorldEvent.action_geo_full_name,
                "event_date",
                WorldEvent.event_date,
                "num_mentions",
                WorldEvent.num_mentions,
                "num_sources",
                WorldEvent.num_sources,
                "source_url",
                WorldEvent.source_url,
                "age_hours",
                func.extract("epoch", func.now() - WorldEvent.event_date) / 3600.0,
            ),
        )
        collection = as_feature_collection(feature)
        cutoff = func.now() - timedelta(days=expiry_days)
        stmt = select(cast(collection, SqlText)).where(WorldEvent.event_date >= cutoff)
        try:
            with Session() as session:
                result = session.scalar(stmt)
                return result if result is not None else EMPTY_FEATURE_COLLECTION
        except Exception as e:
            logger.error(f"Error building world events GeoJSON: {e}")
            return EMPTY_FEATURE_COLLECTION

    def delete_expired(self, expiry_days=7) -> int:
        """Deletes world_event rows older than expiry_days. Returns the number deleted."""
        cutoff = func.now() - timedelta(days=expiry_days)
        stmt = delete(WorldEvent).where(WorldEvent.event_date < cutoff)
        try:
            with Session() as session:
                result = session.execute(stmt)
                session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error deleting expired world events: {e}")
            return 0


class FakeWorldEventAdapter:
    """In-memory fake for world_events, matching WorldEventAdapter's method contracts."""

    def __init__(self):
        self._events: dict[str, dict] = {}

    def upsert_events(self, rows):
        if not rows:
            return
        for r in rows:
            if r.get("lat") is None or r.get("lon") is None:
                continue
            self._events[r["id"]] = {
                "id": r["id"],
                "category": r["category"],
                "event_code": r.get("event_code"),
                "actor1_name": r.get("actor1_name"),
                "actor2_name": r.get("actor2_name"),
                "action_geo_full_name": r.get("action_geo_full_name"),
                "lat": r["lat"],
                "lon": r["lon"],
                "event_date": datetime.fromisoformat(r["event_date"]),
                "num_mentions": r.get("num_mentions"),
                "num_sources": r.get("num_sources"),
                "goldstein_scale": r.get("goldstein_scale"),
                "avg_tone": r.get("avg_tone"),
                "source_url": r.get("source_url"),
            }

    def oldest_event_date(self):
        if not self._events:
            return None
        return min(e["event_date"] for e in self._events.values())

    def get_events_as_geojson(self, expiry_days=7):
        import json

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=expiry_days)
        features = []
        for e in self._events.values():
            if e["event_date"] < cutoff:
                continue
            age_hours = (now - e["event_date"]).total_seconds() / 3600.0
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [e["lon"], e["lat"]]},
                    "properties": {
                        "id": e["id"],
                        "category": e["category"],
                        "event_code": e["event_code"],
                        "actor1_name": e["actor1_name"],
                        "actor2_name": e["actor2_name"],
                        "place": e["action_geo_full_name"],
                        "event_date": e["event_date"].isoformat(),
                        "num_mentions": e["num_mentions"],
                        "num_sources": e["num_sources"],
                        "source_url": e["source_url"],
                        "age_hours": age_hours,
                    },
                }
            )
        return json.dumps({"type": "FeatureCollection", "features": features})

    def delete_expired(self, expiry_days=7) -> int:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=expiry_days)
        expired = [eid for eid, e in self._events.items() if e["event_date"] < cutoff]
        for eid in expired:
            del self._events[eid]
        return len(expired)
