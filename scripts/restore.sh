#!/usr/bin/env bash
# Shared by Makefile (dev) and Makefile.prod's `restore` targets.
set -euo pipefail

DB_SERVICE="${DB_SERVICE:-atmos_gl_db}"
DB_USER="${DB_USER:-agl}"
DB_NAME="${DB_NAME:-atmos_gl}"
DUMP_FILE="${DUMP_FILE:-atmos_gl_dump.sql}"
BACKEND_SERVICES="${BACKEND_SERVICES:-data_collector layer_builder housekeeper map_api}"
COMPOSE_ARGS="${COMPOSE_ARGS:-}"

echo "WARNING: This will DELETE and RECREATE the $DB_NAME database from $DUMP_FILE."
if [ ! -f "$DUMP_FILE" ]; then
    echo "Error: $DUMP_FILE not found."
    exit 1
fi

read -r -p "Are you sure? [y/N] " ans
if [ "${ans:-N}" = "y" ] || [ "${ans:-N}" = "Y" ]; then
    echo "Stopping backend services"
    docker compose $COMPOSE_ARGS stop $BACKEND_SERVICES
    echo "Ensuring $DB_NAME database is running..."
    docker compose $COMPOSE_ARGS up "$DB_SERVICE" -d && sleep 5
    echo "Restoring database..."
    cat "$DUMP_FILE" | docker compose $COMPOSE_ARGS exec -T "$DB_SERVICE" pg_restore -U "$DB_USER" -d postgres --clean --create --if-exists
    echo "Restore complete."
    echo "Stopping $DB_NAME database. Backend left as stopped."
    docker compose $COMPOSE_ARGS stop "$DB_SERVICE"
fi
