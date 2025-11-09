#!/usr/bin/env bash
#
# Stage 3 — run live chaos experiments for a matrix of p_fail values.
#
# Usage:
#   ./scripts/local_stage3_live.sh
# Environment overrides:
#   RUN_ROOT            Directory where per-rate runs are stored (default: ./runs/live)
#   DEMO_REF            Git ref/tag for the demo (default: v2.1.3)
#   P_FAILS             Space-separated list of failure rates (default: "0.1 0.3 0.5 0.7 0.9")
#   CHAOS_WARM_SECONDS  Seconds to wait before injecting faults (default: 60)
#   CHAOS_DURATION      Duration of the fault window (default: 120)
#   SOAK_SECONDS        Total runtime per experiment (default: 240)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/live}"
DEMO_REF="${DEMO_REF:-v2.1.3}"
P_FAILS="${P_FAILS:-0.1 0.3 0.5 0.7 0.9}"
CHAOS_WARM_SECONDS="${CHAOS_WARM_SECONDS:-60}"
CHAOS_DURATION="${CHAOS_DURATION:-120}"
SOAK_SECONDS="${SOAK_SECONDS:-240}"
ENTRYPOINTS_SRC="${ENTRYPOINTS_SRC:-${ROOT}/config/entrypoints.txt}"
TARGETS_SRC="${TARGETS_SRC:-${ROOT}/config/live_targets.yaml}"
ACCOUNTING_HITS="${ACCOUNTING_HITS:-25}"

mkdir -p "${RUN_ROOT}" "${ROOT}/vendor"

if [[ -d "${ROOT}/vendor/opentelemetry-demo/.git" ]]; then
  echo "[stage3] Using existing vendor/opentelemetry-demo checkout"
else
  git clone https://github.com/open-telemetry/opentelemetry-demo.git "${ROOT}/vendor/opentelemetry-demo"
fi

pushd "${ROOT}/vendor/opentelemetry-demo" >/dev/null
git fetch --depth 1 origin "${DEMO_REF}"
git checkout --detach FETCH_HEAD
popd >/dev/null

chaos_services() {
  case "$1" in
    0.1) echo "cartservice" ;;
    0.3) echo "cartservice checkoutservice" ;;
    0.5) echo "cartservice checkoutservice adservice" ;;
    0.7) echo "cartservice checkoutservice adservice recommendationservice" ;;
    0.9) echo "cartservice checkoutservice adservice recommendationservice frauddetectionservice" ;;
    *) echo "" ;;
  esac
}

for rate in ${P_FAILS}; do
  RUN_DIR="${RUN_ROOT}/${rate}"
  mkdir -p "${RUN_DIR}/locust"

  cat > "${ROOT}/vendor/opentelemetry-demo/docker-compose.override.yml" <<YAML
services:
  otel-collector:
    volumes:
      - ${RUN_DIR}/collector:/data
  loadgenerator:
    image: ghcr.io/open-telemetry/demo-loadgenerator:latest
    volumes:
      - ${RUN_DIR}/locust:/data
    command:
      - --autostart
      - --headless
      - --users
      - "50"
      - --spawn-rate
      - "5"
      - --run-time
      - "0"
      - --stop-timeout
      - "90"
      - --host
      - http://frontend:8080
      - --csv
      - /data/locust
      - --csv-full-history
  load-generator:
    image: ghcr.io/open-telemetry/demo-loadgenerator:latest
    volumes:
      - ${RUN_DIR}/locust:/data
    command:
      - --autostart
      - --headless
      - --users
      - "50"
      - --spawn-rate
      - "5"
      - --run-time
      - "0"
      - --stop-timeout
      - "90"
      - --host
      - http://frontend:8080
      - --csv
      - /data/locust
      - --csv-full-history
YAML

  pushd "${ROOT}/vendor/opentelemetry-demo" >/dev/null
  docker compose up --force-recreate --remove-orphans --detach
  popd >/dev/null

  echo "[stage3] (${rate}) Warmup ${CHAOS_WARM_SECONDS}s before injecting faults"
  sleep "${CHAOS_WARM_SECONDS}"

  echo "[stage3] (${rate}) Generating accounting traffic (${ACCOUNTING_HITS} requests)"
  for _ in $(seq 1 "${ACCOUNTING_HITS}"); do
    curl -fsS "http://localhost:8080/api/product-ask-ai-assistant?productId=1" >/dev/null || true
    sleep 1
  done

  services=($(chaos_services "${rate}"))
  if [[ ${#services[@]} -gt 0 ]]; then
    echo "[stage3] (${rate}) Stopping services: ${services[*]}"
    pushd "${ROOT}/vendor/opentelemetry-demo" >/dev/null
    docker compose stop "${services[@]}" || true
    popd >/dev/null
  fi

  echo "[stage3] (${rate}) Holding fault for ${CHAOS_DURATION}s"
  sleep "${CHAOS_DURATION}"

  if [[ ${#services[@]} -gt 0 ]]; then
    echo "[stage3] (${rate}) Restarting services: ${services[*]}"
    pushd "${ROOT}/vendor/opentelemetry-demo" >/dev/null
    docker compose up --detach "${services[@]}" || true
    popd >/dev/null
  fi

  remaining=$(( SOAK_SECONDS - CHAOS_WARM_SECONDS - CHAOS_DURATION ))
  if [[ ${remaining} -gt 0 ]]; then
    echo "[stage3] (${rate}) Cooldown for ${remaining}s"
    sleep "${remaining}"
  fi

  pushd "${ROOT}/vendor/opentelemetry-demo" >/dev/null
  docker compose down -v
  popd >/dev/null

  STATS_FILE="${RUN_DIR}/locust/locust_stats.csv"
  FAIL_FILE="${RUN_DIR}/locust/locust_failures.csv"
  if [[ ! -s "${STATS_FILE}" ]]; then
    echo "[stage3] (${rate}) Missing ${STATS_FILE}; skipping live availability."
    continue
  fi

  python "${ROOT}/scripts/build_live_availability.py" \
    --stats "${STATS_FILE}" \
    --failures "${FAIL_FILE}" \
    --entrypoints "${ENTRYPOINTS_SRC}" \
    --targets "${TARGETS_SRC}" \
    --replica "local-${rate}" \
    --p-grid "${rate}" \
    --out "${RUN_ROOT}/live_availability.csv" \
    --append \
    --no-strict

  echo "[stage3] (${rate}) Appended live availability rows to ${RUN_ROOT}/live_availability.csv"
done

echo "[stage3] Live runs complete. Results under ${RUN_ROOT}"
