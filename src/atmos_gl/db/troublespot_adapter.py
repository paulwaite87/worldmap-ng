#!/usr/bin/env python3
"""Real/Fake adapters for the Troublespots layer (issue #366) -- a derived,
live-computed view over four existing point tables (Earthquakes, Fires, Volcanic
Activity, World Events), not a table of its own. Both adapters bin each source table's
recent rows into a per-cell, per-type row-count map, then hand the derived integer
type-count grid to lib/troublespot_contours.compute_troublespot_bands() -- the SAME
pure function for both, so real-vs-fake parity here only needs to guard the binning/
breakdown step, not re-verify the contour math a second time (already covered by
tests/test_troublespot_contours.py).

The breakdown (which source types, and how many rows of each, contributed to a given
polygon) is derived entirely from the same per-cell counts used for banding -- a cell's
CENTER coordinate is, by construction, one of the exact grid vertices that produced the
contour polygon touching it, so testing "is this cell's center inside this ring"
reliably assigns each qualifying cell to the right polygon even when a band has
multiple disjoint regions. This deliberately avoids a second spatial query against the
raw scattered event points: the smoothed contour polygon covers only a fraction of its
originating cell's true footprint (see this module's git history for the failed first
attempt), so testing raw points against it would silently under-count.

WorldEvent counts as a single type regardless of its own internal category (explosion/
conflict/targeted_violence/diplomacy); splitting it out would let World Events alone
satisfy the convergence minimum -- see the design's roster decision.
"""
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func, select

from atmos_gl.db.engine import Session
from atmos_gl.db.geojson import EMPTY_FEATURE_COLLECTION
from atmos_gl.db.models import Earthquake, Fire, VolcanicActivity, WorldEvent
from atmos_gl.lib.troublespot_contours import BAND_THRESHOLDS, compute_troublespot_bands

logger = logging.getLogger(__name__)

# (type_name, model, time_column) -- the fixed 4-table roster. VolcanicActivity has no
# per-row event timestamp (it's one row per volcano, upserted in place); last_seen_at
# is the "still active" signal every other volcano-liveness check in this codebase
# already uses (Housekeeper's prune_expired_activity), so it's the natural equivalent
# of eq_time/acq_time/event_date here.
_ROSTER = (
    ("earthquake", Earthquake, Earthquake.eq_time),
    ("fire", Fire, Fire.acq_time),
    ("volcanic_activity", VolcanicActivity, VolcanicActivity.last_seen_at),
    ("world_event", WorldEvent, WorldEvent.event_date),
)
_ROSTER_TYPE_NAMES = [name for name, _, _ in _ROSTER]
_THRESHOLD_BY_BAND = dict(BAND_THRESHOLDS)


def _point_in_ring(x: float, y: float, ring: list) -> bool:
    """Standard ray-casting point-in-polygon test (no shapely dependency)."""
    inside = False
    n = len(ring)
    x1, y1 = ring[0]
    for i in range(1, n + 1):
        x2, y2 = ring[i % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def _bands_with_breakdown(cell_type_counts: dict, cell_size_deg: float) -> list:
    """cell_type_counts: {(lat_idx, lon_idx): {type_name: row_count}}. Returns
    [{"band": name, "rings": [{"ring": [(lon, lat), ...], "breakdown": {type_name:
    total_count}}, ...]}, ...]."""
    if not cell_type_counts:
        return []

    lat_idxs = [i for i, _ in cell_type_counts]
    lon_idxs = [j for _, j in cell_type_counts]
    # Pad by one cell on each side so a contour touching the raster's edge still closes.
    lat_idx_min, lat_idx_max = min(lat_idxs) - 1, max(lat_idxs) + 1
    lon_idx_min, lon_idx_max = min(lon_idxs) - 1, max(lon_idxs) + 1
    n_lat = lat_idx_max - lat_idx_min + 1
    n_lon = lon_idx_max - lon_idx_min + 1

    # Grid vertices sit at each cell's CENTER (not its lower-left corner) -- coordinates
    # are derived by multiplying an integer offset by cell_size_deg (never accumulated
    # via repeated addition, e.g. np.arange), so a cell's center is always bit-exactly
    # reproducible from its own index alone.
    lats = [(lat_idx_min + k + 0.5) * cell_size_deg for k in range(n_lat)]
    lons = [(lon_idx_min + k + 0.5) * cell_size_deg for k in range(n_lon)]

    grid = np.zeros((n_lat, n_lon))
    for (i, j), counts in cell_type_counts.items():
        grid[i - lat_idx_min, j - lon_idx_min] = len(counts)

    raw_bands = compute_troublespot_bands(grid, lons, lats)
    cell_centers = {
        (i, j): (lons[j - lon_idx_min], lats[i - lat_idx_min])
        for (i, j) in cell_type_counts
    }

    bands = []
    for band in raw_bands:
        threshold = _THRESHOLD_BY_BAND[band["band"]]
        qualifying = [k for k, counts in cell_type_counts.items() if len(counts) >= threshold]
        rings_out = []
        for ring in band["rings"]:
            breakdown: dict = {}
            for key in qualifying:
                cx, cy = cell_centers[key]
                if _point_in_ring(cx, cy, ring):
                    for type_name, count in cell_type_counts[key].items():
                        breakdown[type_name] = breakdown.get(type_name, 0) + count
            rings_out.append({"ring": ring, "breakdown": breakdown})
        bands.append({"band": band["band"], "rings": rings_out})
    return bands


def _bands_to_geojson(bands: list) -> str:
    features = []
    for band in bands:
        for ring_entry in band["rings"]:
            ring = ring_entry["ring"]
            closed = ring if ring[0] == ring[-1] else [*ring, ring[0]]
            breakdown = ring_entry["breakdown"]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [closed]},
                    "properties": {
                        "band": band["band"],
                        **{t: breakdown.get(t, 0) for t in _ROSTER_TYPE_NAMES},
                    },
                }
            )
    if not features:
        return EMPTY_FEATURE_COLLECTION
    return json.dumps({"type": "FeatureCollection", "features": features})


class TroublespotAdapter:
    """Real adapter: one GROUP BY floor()-binning query per source table, no new table
    of its own."""

    def get_troublespots_as_geojson(self, cell_size_deg=2.0, window_hours=48) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        cell_type_counts: dict = {}
        try:
            with Session() as session:
                for type_name, model, time_col in _ROSTER:
                    lat_idx = func.floor(model.lat / cell_size_deg)
                    lon_idx = func.floor(model.lon / cell_size_deg)
                    stmt = (
                        select(lat_idx, lon_idx, func.count())
                        .where(
                            time_col >= cutoff,
                            model.lat.isnot(None),
                            model.lon.isnot(None),
                        )
                        .group_by(lat_idx, lon_idx)
                    )
                    for row_lat_idx, row_lon_idx, count in session.execute(stmt).all():
                        key = (int(row_lat_idx), int(row_lon_idx))
                        cell_type_counts.setdefault(key, {})[type_name] = int(count)
        except Exception as e:
            logger.error(f"Error binning troublespot source cells: {e}")
            return EMPTY_FEATURE_COLLECTION

        return _bands_to_geojson(_bands_with_breakdown(cell_type_counts, cell_size_deg))


class FakeTroublespotAdapter:
    """In-memory fake, matching TroublespotAdapter's method contract. add_row() is the
    seeding entry point tests use in place of the real adapter's upstream collectors
    (there's no upsert_troublespots -- this layer has no table of its own to seed)."""

    def __init__(self):
        self._rows: dict[str, list[dict]] = {name: [] for name in _ROSTER_TYPE_NAMES}

    def add_row(self, source_type: str, lat: float, lon: float, timestamp: datetime):
        self._rows[source_type].append({"lat": lat, "lon": lon, "timestamp": timestamp})

    def get_troublespots_as_geojson(self, cell_size_deg=2.0, window_hours=48) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        cell_type_counts: dict = {}
        for type_name, rows in self._rows.items():
            for row in rows:
                if row["timestamp"] < cutoff:
                    continue
                key = (
                    int(np.floor(row["lat"] / cell_size_deg)),
                    int(np.floor(row["lon"] / cell_size_deg)),
                )
                counts = cell_type_counts.setdefault(key, {})
                counts[type_name] = counts.get(type_name, 0) + 1

        return _bands_to_geojson(_bands_with_breakdown(cell_type_counts, cell_size_deg))
