#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: pipeline_capture.sh RUN_DIR}
SOAK_SECONDS=${SOAK_SECONDS:-180}
JAEGER_BASE_URL=${JAEGER_BASE_URL:-http://localhost:8080/jaeger}
OTEL_DEMO_DIR=${OTEL_DEMO_DIR:-otel-demo}
TRACE_LIMIT=${TRACE_LIMIT:-1000}

mkdir -p "${RUN_DIR}"

# Ensure the demo is up (idempotent), using its compose file
docker compose -f "${OTEL_DEMO_DIR}/docker-compose.yml" up -d

# Soak to generate traces
sleep "${SOAK_SECONDS}"

# Capture traces via Jaeger (UI is proxied at /jaeger/ui, JSON API at /jaeger/api/*)
python -m edgetyper.capture_oteldemo \
  --base-url "${JAEGER_BASE_URL}" \
  --lookback "30m" \
  --limit "${TRACE_LIMIT}" \
  --out "${RUN_DIR}/spans.jsonl"

# Minimal provenance
{
  echo "target=otel-demo"
  echo "soak_seconds=${SOAK_SECONDS}"
  echo "jaeger_base=${JAEGER_BASE_URL}"
} > "${RUN_DIR}/provenance.txt"

echo "[capture] wrote ${RUN_DIR}/spans.jsonl"
