#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: pipeline_capture.sh RUN_DIR}
SOAK_SECONDS=${SOAK_SECONDS:-180}

# Point at the Envoy front-proxy root; Python will discover the JSON API prefix.
JAEGER_BASE_URL=${JAEGER_BASE_URL:-http://localhost:8080}
OTEL_DEMO_DIR=${OTEL_DEMO_DIR:-otel-demo}
TRACE_LIMIT=${TRACE_LIMIT:-1000}
LOOKBACK=${LOOKBACK:-30m}

mkdir -p "${RUN_DIR}"

# Ensure the demo is up (idempotent)
docker compose -f "${OTEL_DEMO_DIR}/docker-compose.yml" up -d

# Soak to generate traces
sleep "${SOAK_SECONDS}"

# Capture traces via discovered Jaeger JSON API
python -m edgetyper.capture_oteldemo \
  --base-url "${JAEGER_BASE_URL}" \
  --lookback "${LOOKBACK}" \
  --limit "${TRACE_LIMIT}" \
  --out "${RUN_DIR}/spans.jsonl"

# Provenance
{
  echo "target=otel-demo"
  echo "soak_seconds=${SOAK_SECONDS}"
  echo "base_url=${JAEGER_BASE_URL}"
} > "${RUN_DIR}/provenance.txt"

echo "[capture] wrote ${RUN_DIR}/spans.jsonl"
