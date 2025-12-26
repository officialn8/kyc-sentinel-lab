#!/usr/bin/env bash
set -euo pipefail

# Railway-friendly startup:
# - run migrations on boot
# - start uvicorn on $PORT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"

echo "[startup] running alembic migrations..."
alembic upgrade head

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-1}"

echo "[startup] starting uvicorn on ${HOST}:${PORT} (workers=${WORKERS})..."
exec uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --proxy-headers


