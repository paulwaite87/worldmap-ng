#!/usr/bin/env python3
from fastapi import APIRouter, Response, Query, Depends
from atmos_gl.db.world_event_adapter import WorldEventAdapter

router = APIRouter(prefix="/api", tags=["World Events"])


def get_world_event_adapter() -> WorldEventAdapter:
    return WorldEventAdapter()


@router.get("/world_events/geojson")
async def get_world_events_geojson(
    expiry_days: int = Query(7),
    world_event_adapter: WorldEventAdapter = Depends(get_world_event_adapter),
):
    geojson_string = world_event_adapter.get_events_as_geojson(expiry_days)
    return Response(content=geojson_string, media_type="application/json")
