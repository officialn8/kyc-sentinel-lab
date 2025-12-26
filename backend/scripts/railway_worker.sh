#!/usr/bin/env bash
set -euo pipefail

# Worker for durable jobs (see backend/app/worker.py).
# Run as a separate Railway service.

exec python -m app.worker


