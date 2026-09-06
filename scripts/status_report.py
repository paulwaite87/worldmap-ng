#!/usr/bin/env python3
"""Prints a summary of everything currently stored in the database, one section per
data domain -- backs `make status` (both Makefile and Makefile.prod, via scripts/
status.sh). Built against the real SQLAlchemy models (db/models.py) rather than literal
SQL strings, so a column/table rename breaks this at import/query time instead of
silently drifting.

Excludes account/session/system-bookkeeping tables (users, user_settings,
process_status, backfill_requests, viewport_state) -- those aren't collected
map-layer data.
"""
from sqlalchemy import Numeric, cast, func, select

from atmos_gl.db.engine import Session
from atmos_gl.db.models import (
    Aircraft,
    Earthquake,
    FieldCatalog,
    Fire,
    FlightRoute,
    LightningStrike,
    MapRegion,
    Marker,
    Satellite,
    Ship,
    Storm,
    StormTrack,
    VolcanicActivity,
    WorldEvent,
)


def _print_header(title: str) -> None:
    print(f"\n--- {title} ---")


def _print_table(columns: list[str], rows: list[tuple]) -> None:
    if not rows:
        print("(no rows)")
        return
    str_rows = [[str(v) if v is not None else "" for v in row] for row in rows]
    widths = [
        max(len(columns[i]), *(len(r[i]) for r in str_rows)) for i in range(len(columns))
    ]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in str_rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def _region_counts(session, title: str, model, id_col, geom_col) -> None:
    _print_header(title)
    stmt = (
        select(MapRegion.label, func.count(id_col))
        .select_from(MapRegion)
        .outerjoin(model, func.ST_Within(geom_col, MapRegion.boundary))
        .group_by(MapRegion.label)
        .order_by(func.count(id_col).desc())
    )
    _print_table(["region", "count"], session.execute(stmt).all())


def main() -> None:
    with Session() as session:
        _region_counts(session, "Ships Located in Each Region", Ship, Ship.mmsi, Ship.geom)

        _print_header("Database Composition (Unique Ships)")
        stmt = select(
            func.count().filter(Ship.name != "Unknown", Ship.vessel_type != 0).label("full_records"),
            func.count().filter(Ship.name == "Unknown", Ship.vessel_type == 0).label("shadow_records"),
            func.count().label("total"),
        ).select_from(Ship)
        _print_table(["full_records", "shadow_records", "total"], session.execute(stmt).all())

        _region_counts(
            session, "Lightning Strikes in Each Region", LightningStrike, LightningStrike.id, LightningStrike.geom
        )

        _region_counts(session, "Earthquakes in Each Region", Earthquake, Earthquake.id, Earthquake.geom)

        _print_header("Earthquake Magnitude Summary")
        stmt = select(
            func.count().label("total"),
            func.round(cast(func.min(Earthquake.mag), Numeric), 1).label("weakest"),
            func.round(cast(func.max(Earthquake.mag), Numeric), 1).label("strongest"),
            func.round(cast(func.avg(Earthquake.mag), Numeric), 1).label("avg_mag"),
        ).select_from(Earthquake)
        _print_table(["total", "weakest", "strongest", "avg_mag"], session.execute(stmt).all())

        _print_header("Volcanic Activity by Alert Level")
        alert_level = func.coalesce(VolcanicActivity.hans_alert_level, "Not HANS-tracked")
        stmt = (
            select(alert_level.label("alert_level"), func.count())
            .select_from(VolcanicActivity)
            .group_by(alert_level)
            .order_by(func.count().desc())
        )
        _print_table(["alert_level", "count"], session.execute(stmt).all())

        _print_header("Active Fire Detections by Confidence")
        stmt = (
            select(Fire.confidence, func.count())
            .select_from(Fire)
            .group_by(Fire.confidence)
            .order_by(func.count().desc())
        )
        _print_table(["confidence", "count"], session.execute(stmt).all())

        _print_header("World Events by Category")
        stmt = (
            select(WorldEvent.category, func.count())
            .select_from(WorldEvent)
            .group_by(WorldEvent.category)
            .order_by(func.count().desc())
        )
        _print_table(["category", "count"], session.execute(stmt).all())

        _print_header("Active Storms")
        track_points = (
            select(func.count())
            .where(StormTrack.sid == Storm.sid)
            .correlate(Storm)
            .scalar_subquery()
        )
        stmt = select(Storm.sid, Storm.name, track_points.label("track_points")).order_by(Storm.name)
        _print_table(["sid", "name", "track_points"], session.execute(stmt).all())

        _print_header("Satellites Tracked")
        stmt = select(
            func.count().label("total"),
            func.min(Satellite.epoch).label("oldest_epoch"),
            func.max(Satellite.epoch).label("newest_epoch"),
        ).select_from(Satellite)
        _print_table(["total", "oldest_epoch", "newest_epoch"], session.execute(stmt).all())

        _print_header("Weather Model Data by Product")
        stmt = (
            select(
                FieldCatalog.product,
                func.count().label("cached_entries"),
                func.max(FieldCatalog.valid_time).label("latest_valid_time"),
            )
            .select_from(FieldCatalog)
            .group_by(FieldCatalog.product)
            .order_by(FieldCatalog.product)
        )
        _print_table(["product", "cached_entries", "latest_valid_time"], session.execute(stmt).all())

        _print_header("Place Markers by Kind")
        stmt = (
            select(Marker.kind, func.count())
            .select_from(Marker)
            .group_by(Marker.kind)
            .order_by(func.count().desc())
        )
        _print_table(["kind", "count"], session.execute(stmt).all())

        _print_header("Aircraft Currently Tracked")
        stmt = select(
            func.count().filter(Aircraft.on_ground).label("on_ground"),
            func.count().filter(~Aircraft.on_ground).label("airborne"),
            func.count().label("total"),
        ).select_from(Aircraft)
        _print_table(["on_ground", "airborne", "total"], session.execute(stmt).all())

        _print_header("Flight Routes Resolved")
        stmt = select(
            func.count().filter(FlightRoute.stops.is_not(None)).label("with_route"),
            func.count().label("total_lookups"),
        ).select_from(FlightRoute)
        _print_table(["with_route", "total_lookups"], session.execute(stmt).all())


if __name__ == "__main__":
    main()
