#!/usr/bin/env python3
"""Canonical per-layer output file base path, shared by every render task (tasks/*.py,
which write to these paths) and the config API route (routes/config.py, which injects
them into /api/config's response so the frontend's URL construction has exactly one
source of truth, same idea as lib/oisst.py's collector/renderer path sharing).

Not user-configurable (there used to be an editable `outfile` setting; for every
section here except clouds the value a user typed was never the real filename anyway
-- multi-hour/mode-suffix layers only used it as a base to mangle -- so it was removed
from the settings UI). The frontend still reads `cfg.outfile` exactly as before; it's
just computed here and injected by the API now instead of stored in config.json.

Sections with no render task (markers, storms, quakes, volcanoes, lightning) have no
entry -- they're served live via DB-backed API routes, nothing writes a file for them.
"""

OUTFILES = {
    "isobars": "data/isobars.png",
    "wind": "data/wind.png",
    "precipitation": "data/precipitation.png",
    "currents": "data/currents.png",
    "jetstream": "data/jetstream.png",
    "waves": "data/waves.png",
    "temperature": "data/temperature.png",
    "ozone": "data/ozone.png",
    "stormwatch": "data/stormwatch.png",
    "pwat": "data/pwat.png",
    "sst": "data/sst.png",
    "clouds": "data/cloud_map.png",
    "fires": "data/fire_weather.png",
    "greenhouse_gases": "data/greenhouse_gases.png",
    "air_quality": "data/air_quality.png",
    "flood_risk": "data/flood_risk.png",
}
