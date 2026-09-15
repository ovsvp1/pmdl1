#!/bin/sh
set -e

INTERVAL_SECONDS="${PIPELINE_INTERVAL_SECONDS:-300}"

while true; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting pipeline run"
  python /app/code/datasets/data_pipeline.py
  python /app/code/models/train.py
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pipeline run complete. Sleeping ${INTERVAL_SECONDS}s"
  sleep "$INTERVAL_SECONDS"
done
