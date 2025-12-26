#!/usr/bin/env bash
set -euo pipefail

# Run migrations then start the API server.
alembic upgrade head

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"


