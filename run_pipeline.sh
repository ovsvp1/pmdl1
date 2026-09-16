#!/bin/sh
# Runs the full pipeline: data engineering -> model engineering -> deployment.
# Invoked by scheduler.py automatically every 5 minutes.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${SCRIPT_DIR}/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting pipeline run"

"$PYTHON" code/datasets/data_pipeline.py
"$PYTHON" code/models/train.py

# Ensures the API and app are (re)deployed. Since the model is mounted as a
# read-only volume and the API reloads it automatically on change, this is
# idempotent and does not interrupt already-running containers.
docker compose -f code/deployment/docker-compose.yml up -d --build

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pipeline run complete"
