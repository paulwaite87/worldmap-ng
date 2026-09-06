#!/usr/bin/env python3
import logging
import os
import secrets
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

# Import the new decoupled router files
from atmos_gl.routes import (
    auth,
    satellites,
    storms,
    volcanoes,
    quakes,
    fires,
    world_events,
    troublespots,
    lightning,
    shipping,
    config,
    me_settings,
    terminator,
    backfill,
    markers,
    status,
    system_status,
    flightradar,
    layer_builder,
    layer_availability,
    version,
)
from atmos_gl.lib.config import AtmosGLConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="Atmos GL Configuration API")

# uvicorn logs every request at INFO regardless of our own log_level setting -- only
# let those through when common.log_level is explicitly DEBUG, same threshold every
# other service uses to decide "chatty" vs "quiet". Runs once at import (uvicorn
# configures its own logging before importing the app, so nothing resets this after).
_config_path = os.getenv("CONFIG_PATH", "./config/atmos-gl.json")
if os.path.exists(_config_path):
    _live_config = AtmosGLConfig(_config_path)
    _live_config.load()
    if _live_config.get_setting("common", "log_level", "INFO") != "DEBUG":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# #301 unified map_ui + map_api behind one origin, so browser calls through nginx are
# same-origin and never hit this middleware at all -- this only governs a legitimate
# cross-origin caller, and now that /api/auth/me's session cookie carries real identity
# (issue #303), it must name real origins rather than reflect-any-origin ("*" +
# allow_credentials=True doesn't literally send "*" -- Starlette reflects the request's
# actual Origin back instead, which amounts to allowing every origin). PUBLIC_ORIGIN lets
# a deployment add its real origin(s) (comma-separated); defaults to local dev's origin so
# `make up` works out of the box.
origins = [o.strip() for o in os.getenv("PUBLIC_ORIGIN", "http://localhost:8180").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Required by authlib's Starlette OAuth client (routes/auth.py) to hold Google
# sign-in state/nonce across the redirect round-trip -- NOT the app's own session
# (that's a separate, DB-backed, long-lived cookie; see lib/auth.py). Falls back to a
# per-process random secret if SESSION_SECRET isn't set, so this is always genuinely
# signed rather than silently using a predictable/empty key -- the one cost is that an
# in-flight Google sign-in doesn't survive a container restart, which just means the
# visitor retries.
_session_secret = os.getenv("SESSION_SECRET")
if not _session_secret:
    logger.warning(
        "SESSION_SECRET not set; using an ephemeral per-process secret "
        "(a Google sign-in in progress won't survive a restart)."
    )
    _session_secret = secrets.token_urlsafe(32)
app.add_middleware(SessionMiddleware, secret_key=_session_secret)

# Keep the static asset pipeline mounted at root. Docker's bind mount
# (./data:/opt/project/data) auto-creates this directory at container start, but a bare
# `uv run pytest`/uvicorn invocation (e.g. CI) has no such mount -- StaticFiles requires
# the directory to exist at import time, so ensure it does rather than crash on import.
os.makedirs("data", exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")

# -------------------------------------------------------------
# ROUTER HOOKS - Registering the modular layout blocks
# -------------------------------------------------------------
app.include_router(auth.router)
app.include_router(terminator.router)
app.include_router(satellites.router)
app.include_router(storms.router)
app.include_router(volcanoes.router)
app.include_router(quakes.router)
app.include_router(fires.router)
app.include_router(world_events.router)
app.include_router(troublespots.router)
app.include_router(lightning.router)
app.include_router(shipping.router)
app.include_router(config.router)
app.include_router(config.ui_router)
app.include_router(me_settings.router)
app.include_router(me_settings.ui_router)
app.include_router(markers.router)
app.include_router(backfill.router)
app.include_router(status.router)
app.include_router(system_status.router)
app.include_router(flightradar.router)
app.include_router(layer_builder.router)
app.include_router(layer_availability.router)
app.include_router(version.router)


@app.get("/")
def health_check():
    return {"status": "online", "message": "Atmos GL Engine routing operational."}
