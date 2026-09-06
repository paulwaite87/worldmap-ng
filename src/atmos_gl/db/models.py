from datetime import date, datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    REAL,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


AisVesselTier = Enum("A", "B", name="ais_vessel_tier")


class Ship(Base):
    __tablename__ = "ships"
    __table_args__ = (
        Index("idx_ships_geom", "geom", postgresql_using="gist"),
        Index("idx_ships_last_update", "last_position_update"),
    )

    mmsi: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    vessel_type: Mapped[int | None] = mapped_column(Integer)
    imo: Mapped[int | None] = mapped_column(Integer)
    callsign: Mapped[str | None] = mapped_column(String(50))
    draught: Mapped[float | None] = mapped_column(Numeric(5, 2))
    prev_draught: Mapped[float | None] = mapped_column(Numeric(5, 2), default=0.0)
    length: Mapped[int | None] = mapped_column(Integer)
    beam: Mapped[int | None] = mapped_column(Integer)
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp()
    )
    lat: Mapped[float | None] = mapped_column()
    lon: Mapped[float | None] = mapped_column()
    nav_status: Mapped[int | None] = mapped_column(Integer, default=0)
    sog: Mapped[float | None] = mapped_column(default=0.0)
    cog: Mapped[float | None] = mapped_column(default=0.0)
    last_position_update: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    destination: Mapped[str | None] = mapped_column(Text)
    vessel_class: Mapped[str | None] = mapped_column(String(50))
    ais_tier: Mapped[str] = mapped_column(AisVesselTier, nullable=False, default="A")


class ShipPosition(Base):
    __tablename__ = "ship_position"
    __table_args__ = (
        Index("idx_ship_pos_geom", "geom", postgresql_using="gist"),
        Index("idx_ship_pos_mmsi", "mmsi"),
        Index("idx_ship_pos_time", "acquired_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    mmsi: Mapped[str] = mapped_column(
        String(20), ForeignKey("ships.mmsi", ondelete="CASCADE"), nullable=False
    )
    lat: Mapped[float | None] = mapped_column()
    lon: Mapped[float | None] = mapped_column()
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    sog: Mapped[float | None] = mapped_column(default=0.0)
    cog: Mapped[float | None] = mapped_column(default=0.0)
    nav_status: Mapped[int | None] = mapped_column(Integer, default=0)
    acquired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MapRegion(Base):
    __tablename__ = "map_region"
    # Index/sequence/constraint names below are stale ("ship_regions_*") from a table
    # rename that never touched them — kept as-is to match the live schema exactly.
    __table_args__ = (Index("idx_ship_regions_boundary", "boundary", postgresql_using="gist"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str | None] = mapped_column(String(100), unique=True)
    boundary: Mapped[str | None] = mapped_column(
        Geometry("POLYGON", srid=4326, spatial_index=False)
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp()
    )


class LightningStrike(Base):
    __tablename__ = "lightning_strikes"
    __table_args__ = (
        Index("idx_lightning_geom", "geom", postgresql_using="gist"),
        Index("idx_lightning_time", "acquired_at"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    lat: Mapped[float | None] = mapped_column(REAL)
    lon: Mapped[float | None] = mapped_column(REAL)
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    quality: Mapped[str | None] = mapped_column(Text)
    acquired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Earthquake(Base):
    __tablename__ = "earthquakes"
    __table_args__ = (
        Index("idx_quakes_geom", "geom", postgresql_using="gist"),
        Index("idx_quakes_time", "eq_time"),
    )

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    mag: Mapped[float | None] = mapped_column(REAL)
    depth: Mapped[float | None] = mapped_column(REAL)
    place: Mapped[str | None] = mapped_column(Text)
    eq_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lat: Mapped[float | None] = mapped_column()
    lon: Mapped[float | None] = mapped_column()
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))


class VolcanicActivity(Base):
    """Current-state volcanic activity, one row per volcano, upserted from two live
    sources joined on the shared Smithsonian VNUM (confirmed identical scheme in both):
    the GVP Weekly Volcanic Activity Report (GeoRSS, global base -- name/country/lat/
    lon/activity_type/report_description) and USGS HANS getElevatedVolcanoes (US-only
    enrichment -- hans_*). Replaces the old static NOAA HazEL historical-catalog
    `Volcano` model entirely (see issue #253) -- no history table, no VEI/significant/
    date-code filtering; liveness is last_seen_at-driven (Housekeeper prunes stale
    rows), not a fixed catalog.

    lat/lon/geom/name/country persist across polls even when a poll's upsert carries
    no fresh GVP sighting (a HANS-elevated volcano absent from this week's GVP report)
    -- see VolcanicActivityAdapter.upsert_activity's ON CONFLICT semantics."""

    __tablename__ = "volcanic_activity"
    __table_args__ = (Index("idx_volcanic_activity_geom", "geom", postgresql_using="gist"),)

    vnum: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(Text)
    lat: Mapped[float | None] = mapped_column()
    lon: Mapped[float | None] = mapped_column()
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    activity_type: Mapped[str | None] = mapped_column(Text)
    report_description: Mapped[str | None] = mapped_column(Text)
    hans_color_code: Mapped[str | None] = mapped_column(String(10))
    hans_alert_level: Mapped[str | None] = mapped_column(String(20))
    hans_notice_url: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Fire(Base):
    """NASA FIRMS VIIRS_SNPP_NRT active-fire detections. id is synthesised
    (satellite|lat|lon|acq_date|acq_time) -- FIRMS assigns no detection-level id of its
    own, unlike USGS quakes/NOAA volcanoes."""

    __tablename__ = "fires"
    __table_args__ = (
        Index("idx_fires_geom", "geom", postgresql_using="gist"),
        Index("idx_fires_acq_time", "acq_time"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    lat: Mapped[float | None] = mapped_column(REAL)
    lon: Mapped[float | None] = mapped_column(REAL)
    brightness: Mapped[float | None] = mapped_column(REAL)
    frp: Mapped[float | None] = mapped_column(REAL)
    confidence: Mapped[str | None] = mapped_column(String(10))
    satellite: Mapped[str | None] = mapped_column(String(20))
    daynight: Mapped[str | None] = mapped_column(String(1))
    acq_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))


class Storm(Base):
    __tablename__ = "storms"

    sid: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(50))
    cone_geom: Mapped[str | None] = mapped_column(
        Geometry("POLYGON", srid=4326, spatial_index=False)
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class StormTrack(Base):
    __tablename__ = "storm_track"
    __table_args__ = (Index("idx_storm_track_sid", "sid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sid: Mapped[str | None] = mapped_column(
        String(10), ForeignKey("storms.sid", ondelete="CASCADE")
    )
    record_type: Mapped[str | None] = mapped_column(String(10))
    dt: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    tau: Mapped[int | None] = mapped_column(Integer)
    lat: Mapped[float | None] = mapped_column(Numeric)
    lon: Mapped[float | None] = mapped_column(Numeric)
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    # ATCF b-deck/a-deck fields 8/9/10 (VMAX/MSLP/TY) -- present on every track/forecast
    # line already parsed for LAT/LON/TIME, just not previously captured. wind_kt is
    # knots and pressure_hpa is hPa (ATCF's own units, unconverted); category is the raw
    # ATCF storm-type code (e.g. "TD"/"TS"/"HU") -- translated to friendly English only
    # at render time (ui/modules/storms.js), so the stored value stays reusable for
    # anything else that might want the raw code later (e.g. colour-coding by category).
    wind_kt: Mapped[int | None] = mapped_column(Integer)
    pressure_hpa: Mapped[int | None] = mapped_column(Integer)
    category: Mapped[str | None] = mapped_column(String(4))


class Satellite(Base):
    __tablename__ = "satellites"
    __table_args__ = (Index("idx_satellites_name", "name"),)

    norad_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120))
    omm: Mapped[dict] = mapped_column(JSONB, nullable=False)
    epoch: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class FieldCatalog(Base):
    __tablename__ = "field_catalog"
    __table_args__ = (
        Index("idx_field_catalog_product", "product"),
        Index("idx_field_catalog_updated", text("updated_at DESC")),
    )

    run_date: Mapped[date] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(2), primary_key=True)
    fhour: Mapped[int] = mapped_column(Integer, primary_key=True)
    product: Mapped[str] = mapped_column(String(32), primary_key=True)
    nlat: Mapped[int | None] = mapped_column(Integer)
    nlon: Mapped[int | None] = mapped_column(Integer)
    valid_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    storage_uri: Mapped[str | None] = mapped_column(String(256))


class BackfillRequest(Base):
    __tablename__ = "backfill_requests"
    __table_args__ = (Index("idx_backfill_status", "status"),)

    run_date: Mapped[date] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(2), primary_key=True)
    fhour: Mapped[int] = mapped_column(Integer, primary_key=True)
    product: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="requested")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ProcessStatus(Base):
    __tablename__ = "process_status"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    last_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    # "idle" | "running" | "success" | "failed" -- set by record_process_start()/
    # record_process_run() (db/process_status_adapter.py). started_at is cleared back
    # to NULL on completion (success or failure); it's only meaningful while status is
    # "running".
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="idle")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # health/health_detail/health_at: a transient upstream-condition signal (e.g.
    # "rate_limited"/"blocked"), set ONLY by record_health() -- deliberately
    # independent of status/last_error above, since a handful of throttled requests
    # doesn't mean the collector's whole run failed (see record_health()'s docstring).
    # Read-time TTL expiry (lib/data_status.py's read_health_status()) treats an
    # unrenewed condition as resolved rather than showing it stale forever, so nothing
    # ever explicitly clears these back to NULL in the common case.
    health: Mapped[str | None] = mapped_column(String(20))
    health_detail: Mapped[str | None] = mapped_column(Text)
    health_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # progress_current/progress_total: a generic "N of M units done" counter, set ONLY
    # by record_progress() -- issue #215's Flight Radar hotspot-coverage bar is the
    # first user (AircraftCollector reports how many of the currently-prioritized
    # viewport cells have been sampled), but the shape is generic enough for any
    # future collector wanting a similar Data Status progress bar. Deliberately
    # independent of status/last_error/health above, same reasoning as those.
    progress_current: Mapped[int | None] = mapped_column(Integer)
    progress_total: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Marker(Base):
    __tablename__ = "markers"
    __table_args__ = (Index("idx_markers_kind_priority", "kind", "priority"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False, default="place")
    country: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int | None] = mapped_column(Integer)
    pop: Mapped[int | None] = mapped_column(BigInteger)
    capital: Mapped[bool | None] = mapped_column(Boolean)
    color: Mapped[str | None] = mapped_column(Text)
    timezone: Mapped[str | None] = mapped_column(Text)
    lat: Mapped[float] = mapped_column(nullable=False)
    lon: Mapped[float] = mapped_column(nullable=False)
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    wx_temp_c: Mapped[float | None] = mapped_column(REAL)
    wx_humidity_pct: Mapped[float | None] = mapped_column(REAL)
    wx_wind_ms: Mapped[float | None] = mapped_column(REAL)
    wx_wind_dir_deg: Mapped[float | None] = mapped_column(REAL)
    wx_valid_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    wx_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Aircraft(Base):
    """Flight Radar master record (issue #215) -- keyed on the ICAO 24-bit address
    ("hex", the aircraft equivalent of Ship.mmsi), mirroring Ship's split of static
    identity fields alongside a redundant current-variable-state snapshot (see
    AircraftTrack for the append-only history of the same variable fields)."""

    __tablename__ = "aircraft"
    __table_args__ = (
        Index("idx_aircraft_geom", "geom", postgresql_using="gist"),
        Index("idx_aircraft_last_seen", "last_seen"),
    )

    hex: Mapped[str] = mapped_column(String(12), primary_key=True)
    registration: Mapped[str | None] = mapped_column(String(20))
    aircraft_type: Mapped[str | None] = mapped_column(String(10))
    # ADS-B emitter category (adsb.lol's `category`, e.g. "A3"/"B2"/"C1") -- the
    # frontend needs this raw code for ADR 0008's ground-vehicle filter (C*) and
    # glider-icon selection (B*), not just aircraft_type.
    category: Mapped[str | None] = mapped_column(String(4))
    flight: Mapped[str | None] = mapped_column(String(20))
    lat: Mapped[float | None] = mapped_column()
    lon: Mapped[float | None] = mapped_column()
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    alt_baro_ft: Mapped[float | None] = mapped_column(REAL)
    on_ground: Mapped[bool | None] = mapped_column(Boolean, default=False)
    gs: Mapped[float | None] = mapped_column(REAL)
    track: Mapped[float | None] = mapped_column(REAL)
    baro_rate: Mapped[float | None] = mapped_column(REAL)
    nav_altitude_mcp: Mapped[float | None] = mapped_column(REAL)
    squawk: Mapped[str | None] = mapped_column(String(4))
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp()
    )
    last_position_update: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AircraftTrack(Base):
    """Append-only Flight Radar history (issue #215) -- deliberately NOT named
    "AircraftPosition": ShipPosition already demonstrates that name is a misnomer once
    a table carries speed/course/status alongside position, so this one is named for
    what it actually is from the start (renaming ShipPosition itself is a separate,
    out-of-scope future fix). One row per sample the collector records; pruned only by
    Housekeeper.prune_aircraft_tracks, never inline by the collector."""

    __tablename__ = "aircraft_track"
    __table_args__ = (
        Index("idx_aircraft_track_geom", "geom", postgresql_using="gist"),
        Index("idx_aircraft_track_hex", "hex"),
        Index("idx_aircraft_track_time", "acquired_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    hex: Mapped[str] = mapped_column(
        String(12), ForeignKey("aircraft.hex", ondelete="CASCADE"), nullable=False
    )
    lat: Mapped[float | None] = mapped_column()
    lon: Mapped[float | None] = mapped_column()
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    alt_baro_ft: Mapped[float | None] = mapped_column(REAL)
    on_ground: Mapped[bool | None] = mapped_column(Boolean, default=False)
    gs: Mapped[float | None] = mapped_column(REAL)
    track: Mapped[float | None] = mapped_column(REAL)
    baro_rate: Mapped[float | None] = mapped_column(REAL)
    category: Mapped[str | None] = mapped_column(String(4))
    squawk: Mapped[str | None] = mapped_column(String(4))
    acquired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AircraftInterest(Base):
    """One row per active Flight Radar viewer/session (issue #215) -- the DB-mediated
    'hotspot' signal AircraftCollector reads each cycle to prioritize whichever areas
    have an active viewer. Written as a side effect of the frontend's own viewport-read
    requests (upsert-by-viewer_id, keyed on a client-generated session id -- not yet
    wired up; that's the REST-route half of issue #215, still to come). A row not
    refreshed within the collector's configured max-age is treated as gone, same as a
    disconnected RegionManager subscriber."""

    __tablename__ = "aircraft_interest"

    viewer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    west: Mapped[float] = mapped_column(nullable=False)
    south: Mapped[float] = mapped_column(nullable=False)
    east: Mapped[float] = mapped_column(nullable=False)
    north: Mapped[float] = mapped_column(nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FlightRoute(Base):
    """Callsign -> route lookup (issue #215's route-enrichment follow-on), resolved via
    adsb.lol's routeset endpoint (data_collector.datasources.flightradar_routeset).

    Keyed on `flight` (the ICAO callsign, e.g. "ANZ423"), NOT on `hex` -- a route is a
    fact about the SCHEDULED FLIGHT NUMBER, not the physical airframe that happens to
    be operating it today, so this is deliberately independent of Aircraft/AircraftTrack
    (no FK, no join key back to hex at all). A single row here serves every aircraft
    that has ever flown this callsign, rather than one lookup per hex.

    `stops` is the ordered `_airports` list from the routeset response verbatim (origin
    first, destination last, any technical/intermediate stop(s) in between) -- a JSONB
    array rather than a child table, since it's always read/written as one atomic unit
    alongside its parent row and never queried or updated independently (unlike
    AircraftTrack, which genuinely needs per-row inserts and time-based pruning). NULL
    when adsb.lol has no route for this callsign (GA/charter/military/unlisted).

    `plausible` is adsb.lol's own great-circle sanity check (the aircraft's real
    position at lookup time against the route's path) -- NULL when there's no match to
    check, True/False when there is.

    `checked_at` drives staleness uniformly for both matches and non-matches (see
    flight_route_adapter.py) -- no separate per-column invalidation logic is needed
    here, since keying by callsign (not hex) means there's no "route resolved for a
    stale callsign" case to detect: whichever callsign an aircraft is currently
    squawking is simply looked up fresh against this table.

    No Housekeeper pruning: unlike AircraftTrack's continuously-growing sighting
    history, this is a small, slowly-changing reference table (at most one row per
    distinct callsign ever seen), refreshed in place rather than deleted."""

    __tablename__ = "flight_route"

    flight: Mapped[str] = mapped_column(String(20), primary_key=True)
    stops: Mapped[list | None] = mapped_column(JSONB)
    plausible: Mapped[bool | None] = mapped_column(Boolean)
    checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ViewportState(Base):
    """Single-row table holding the most recently reported map viewport (see
    ViewportAdapter) -- data_collector's LightningCollector runs in a separate
    container from map_api, so this is the only channel between "what the frontend is
    currently looking at" and the collector's scan prioritization; an in-memory flag
    in map_api wouldn't be visible there. id is always "current"."""

    __tablename__ = "viewport_state"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    lat: Mapped[float] = mapped_column(REAL, nullable=False)
    lon: Mapped[float] = mapped_column(REAL, nullable=False)
    zoom: Mapped[float | None] = mapped_column(REAL)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class User(Base):
    """A signed-in visitor (issue #303). is_admin is refreshed from ADMIN_EMAILS on
    every login (UserAdapter.get_or_create_user) -- not re-checked on every request --
    so revoking admin access takes effect the next time that user logs in again."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UserIdentity(Base):
    """One row per (user, sign-in provider) -- kept separate from User itself so a
    second/third provider (e.g. GitHub, Microsoft) is a new row here, never a schema
    change or a new column on User. provider_user_id is that provider's own stable
    subject id (Google's `sub` claim), not the email -- an email can change or be
    reused, the provider's subject id doesn't."""

    __tablename__ = "user_identities"
    __table_args__ = (
        UniqueConstraint(
            "provider", "provider_user_id", name="uq_user_identities_provider_subject"
        ),
        Index("idx_user_identities_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UserSession(Base):
    """A signed-in session (issue #303). token is the opaque value stored in the
    browser's session cookie -- DB-backed (not a signed/stateless token) specifically
    so a session can be revoked (deleted here) without waiting for client-side expiry,
    e.g. on logout or a future account ban."""

    __tablename__ = "user_sessions"
    __table_args__ = (Index("idx_user_sessions_user_id", "user_id"),)

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class UserSettings(Base):
    """Per-user personalized view-settings overrides (issue #305/#314) -- one row per
    user, `overrides` a sparse {section: {option: value}} map of only what that user
    has explicitly changed from the site's global default. Merged on top of
    AtmosGLConfig's own settings by routes/config.py's GET /api/config whenever a
    valid session is present; an anonymous request never touches this table at all."""

    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    overrides: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class WorldEvent(Base):
    """A GDELT Event Database 2.0 record matching the curated CAMEO code allowlist
    (see collectors/world_events.py) -- explosions, warfare, targeted/mass violence, or
    a high-level diplomatic meeting. id is GDELT's own GLOBALEVENTID, so re-ingesting
    the same event (a backfill re-covering a window collect() already filled) is a
    plain no-op upsert, never a duplicate row."""

    __tablename__ = "world_events"
    __table_args__ = (
        Index("idx_world_events_geom", "geom", postgresql_using="gist"),
        Index("idx_world_events_event_date", "event_date"),
    )

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    event_code: Mapped[str | None] = mapped_column(String(10))
    actor1_name: Mapped[str | None] = mapped_column(Text)
    actor2_name: Mapped[str | None] = mapped_column(Text)
    action_geo_full_name: Mapped[str | None] = mapped_column(Text)
    lat: Mapped[float | None] = mapped_column(REAL)
    lon: Mapped[float | None] = mapped_column(REAL)
    geom: Mapped[str | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    event_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    num_mentions: Mapped[int | None] = mapped_column(Integer)
    num_sources: Mapped[int | None] = mapped_column(Integer)
    goldstein_scale: Mapped[float | None] = mapped_column(REAL)
    avg_tone: Mapped[float | None] = mapped_column(REAL)
    source_url: Mapped[str | None] = mapped_column(Text)
