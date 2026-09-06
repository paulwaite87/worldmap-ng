#!/usr/bin/env bash
# Shared by Makefile (dev) and Makefile.prod's `backup` targets.
set -euo pipefail

DB_SERVICE="${DB_SERVICE:-atmos_gl_db}"
DB_USER="${DB_USER:-agl}"
DB_NAME="${DB_NAME:-atmos_gl}"
DUMP_FILE="${DUMP_FILE:-atmos_gl_dump.sql}"
COMPOSE_ARGS="${COMPOSE_ARGS:-}"

echo "Ensuring $DB_NAME database is running"
docker compose $COMPOSE_ARGS up "$DB_SERVICE" -d
echo "Creating compressed database backup to $DUMP_FILE..."
docker compose $COMPOSE_ARGS exec "$DB_SERVICE" pg_dump -U "$DB_USER" -Fc "$DB_NAME" > "$DUMP_FILE"
echo "Backup complete."
