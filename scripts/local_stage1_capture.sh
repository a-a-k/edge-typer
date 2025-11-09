#!/usr/bin/env bash
#
# Stage 1 — capture traces and Locust stats from the OpenTelemetry Demo.
#
# Usage:
#   ./scripts/local_stage1_capture.sh
# Environment overrides:
#   RUN_ROOT        Directory where run artifacts are written (default: ./runs/capture)
#   DEMO_REF        Git ref/tag of opentelemetry-demo (default: v2.1.3)
#   SOAK_SECONDS    How long to keep the demo running (default: 900s)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/capture}"
DEMO_REF="${DEMO_REF:-v2.1.3}"
SOAK_SECONDS="${SOAK_SECONDS:-900}"
RUN_DIR="${RUN_ROOT}/$(date +%Y%m%d-%H%M%S)"

mkdir -p "${RUN_DIR}/collector" "${ROOT}/vendor"

if [[ -d "${ROOT}/vendor/opentelemetry-demo/.git" ]]; then
  echo "[stage1] Found existing vendor/opentelemetry-demo checkout"
else
  git clone https://github.com/open-telemetry/opentelemetry-demo.git "${ROOT}/vendor/opentelemetry-demo"
fi

pushd "${ROOT}/vendor/opentelemetry-demo" >/dev/null
git fetch --depth 1 origin "${DEMO_REF}"
git checkout --detach FETCH_HEAD

cat > src/otel-collector/otelcol-config-extras.yml <<'YAML'
exporters:
  file/edgetyper:
    path: /data/otel-traces.json
service:
  pipelines:
    traces:
      exporters: [spanmetrics, file/edgetyper]
YAML

cat > docker-compose.override.yml <<YAML
services:
  otel-collector:
    volumes:
      - ${RUN_DIR}/collector:/data
  loadgenerator:
    image: ghcr.io/open-telemetry/demo-loadgenerator:latest
  load-generator:
    image: ghcr.io/open-telemetry/demo-loadgenerator:latest
YAML

docker compose up --force-recreate --remove-orphans --detach
popd >/dev/null

echo "[stage1] Waiting up to 3 minutes for services to become healthy..."
for _ in $(seq 1 36); do
  unhealthy=$(docker ps --format '{{.Names}} {{.Status}}' | grep -E 'unhealthy|restarting' || true)
  if [[ -z "${unhealthy}" ]]; then
    break
  fi
  sleep 5
done

echo "[stage1] Soaking for ${SOAK_SECONDS}s..."
sleep "${SOAK_SECONDS}"

pushd "${ROOT}/vendor/opentelemetry-demo" >/dev/null
docker compose down -v
popd >/dev/null

echo "TRACES_JSON=${RUN_DIR}/collector/otel-traces.json" > "${RUN_DIR}/capture.env"
echo "[stage1] Traces stored at ${RUN_DIR}/collector/otel-traces.json"
