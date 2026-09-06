#!/usr/bin/env bash
# Shared by Makefile (dev) and Makefile.prod's `status` targets -- runs the real
# ORM-backed report (status_report.py) inside a container that already has the app's
# Python environment, rather than each Makefile hand-maintaining its own copy of
# literal SQL. This whole scripts/ directory exists because Makefile.prod had drifted
# from Makefile silently (its own copies of bootstrap-config/status never picked up
# fixes made to the dev Makefile) -- one canonical script per shared action now,
# called identically by both Makefiles, so that can't happen again.
set -euo pipefail

BUILDER_SERVICE="${BUILDER_SERVICE:-layer_builder}"
COMPOSE_ARGS="${COMPOSE_ARGS:-}"

# No "ensure services up" step -- the original raw-SQL status target never had one
# either (it assumed the stack was already running), and adding one here would force
# a docker-compose build-context re-verification on every `make status` in dev (prod's
# pre-built-image compose file doesn't have this cost, but dev's does).
docker compose $COMPOSE_ARGS exec -T "$BUILDER_SERVICE" uv run python scripts/status_report.py
