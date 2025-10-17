#!/usr/bin/env bash
set -euo pipefail

DEMO_REF="${DEMO_REF:-main}"
SOAK_SECONDS="${SOAK_SECONDS:-180}"
RUN_DIR="${RUN_DIR:-$PWD/runs/$(date +%s)}"
mkdir -p "${RUN_DIR}/collector" vendor

# 1) Clone & pin the OTel Demo
if [ -d vendor/opentelemetry-demo/.git ]; then
  echo "[capture] Using existing vendor/opentelemetry-demo checkout"
else
  git clone --depth 1 --branch "${DEMO_REF}" https://github.com/open-telemetry/opentelemetry-demo.git vendor/opentelemetry-demo
fi

# 2) Create extras config (file exporter) and Compose override (mount /data)
cat > vendor/opentelemetry-demo/src/otel-collector/otelcol-config-extras.yml <<'YAML'
exporters:
  file/edgetyper:
    path: /data/otel-traces.json
service:
  pipelines:
    traces:
      exporters: [spanmetrics, file/edgetyper]
YAML

cat > vendor/opentelemetry-demo/docker-compose.override.yml <<YAML
services:
  otelcol:
    volumes:
      - ${RUN_DIR}/collector:/data
YAML

# 3) Bring up, wait for unhealthy/restarting to clear, soak, then down
pushd vendor/opentelemetry-demo >/dev/null
docker compose up --force-recreate --remove-orphans --detach
popd >/dev/null

echo "[capture] Waiting up to 3 minutes for services to become healthy…"
for i in {1..36}; do
  bad=$(docker ps --format '{{.Names}} {{.Status}}' | grep -E 'unhealthy|restarting' || true)
  if [ -z "$bad" ]; then break; fi
  sleep 5
done

echo "[capture] Soaking for ${SOAK_SECONDS}s…"
sleep "${SOAK_SECONDS}"

pushd vendor/opentelemetry-demo >/dev/null
docker compose down -v
popd >/dev/null

echo "TRACES_JSON=${RUN_DIR}/collector/otel-traces.json" > "${RUN_DIR}/capture.env"
echo "[capture] Wrote ${RUN_DIR}/collector/otel-traces.json"
