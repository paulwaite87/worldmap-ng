#!/usr/bin/env bash
# Shared by Makefile (dev) and Makefile.prod's `psql` targets.
set -euo pipefail

DB_SERVICE="${DB_SERVICE:-atmos_gl_db}"
DB_USER="${DB_USER:-agl}"
DB_NAME="${DB_NAME:-atmos_gl}"
COMPOSE_ARGS="${COMPOSE_ARGS:-}"

echo "Ensuring $DB_NAME database is running"
docker compose $COMPOSE_ARGS up "$DB_SERVICE" -d
docker compose $COMPOSE_ARGS exec "$DB_SERVICE" psql -U "$DB_USER" "$DB_NAME"
