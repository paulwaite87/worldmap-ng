#!/usr/bin/env python3
from fastapi import APIRouter, Response, Query, Depends
from atmos_gl.db.troublespot_adapter import TroublespotAdapter

router = APIRouter(prefix="/api", tags=["Troublespots"])


def get_troublespot_adapter() -> TroublespotAdapter:
    return TroublespotAdapter()


@router.get("/troublespots/geojson")
async def get_troublespots_geojson(
    cell_size_deg: float = Query(2.0),
    window_hours: int = Query(48),
    troublespot_adapter: TroublespotAdapter = Depends(get_troublespot_adapter),
):
    geojson_string = troublespot_adapter.get_troublespots_as_geojson(cell_size_deg, window_hours)
    return Response(content=geojson_string, media_type="application/json")
