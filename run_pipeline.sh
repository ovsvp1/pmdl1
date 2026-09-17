#!/bin/sh
# Start the complete Dockerized MLOps system.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
docker compose -f code/deployment/docker-compose.yml up -d --build
