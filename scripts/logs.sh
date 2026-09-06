#!/usr/bin/env bash
# Shared by Makefile (dev) and Makefile.prod's `logs` targets. Pass a service name as
# $1 to tail just that service; omit it to tail everything.
set -euo pipefail

COMPOSE_ARGS="${COMPOSE_ARGS:-}"

if [ -z "${1:-}" ]; then
    docker compose $COMPOSE_ARGS logs -f
else
    docker compose $COMPOSE_ARGS logs -f "$1"
fi
