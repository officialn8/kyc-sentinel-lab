#!/usr/bin/env bash
set -euo pipefail

# Railway-friendly startup:
# - run migrations on boot
# - start uvicorn on $PORT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add current directory to Python path
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "[startup] running alembic migrations..."
alembic upgrade head

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-1}"

# Ensure OpenCV system dependencies are available (fixes libGL errors on Railway)
if ! ldconfig -p | grep -q "libGL.so.1"; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    echo "[startup] installing OpenCV runtime libraries..."
    apt-get update -qq
    apt-get install -y -qq libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
  else
    echo "[startup] warning: apt-get not available to install libGL dependencies" >&2
  fi
fi

echo "[startup] starting uvicorn on ${HOST}:${PORT} (workers=${WORKERS})..."
exec uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --proxy-headers


