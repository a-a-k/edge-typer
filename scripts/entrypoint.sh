#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${RUN_ID:-${GITHUB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}}
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[entrypoint] RUN_DIR=${RUN_DIR}"
echo "[entrypoint] OTEL_DEMO_DIR=${OTEL_DEMO_DIR:-unset}"
echo "[entrypoint] JAEGER_BASE_URL=${JAEGER_BASE_URL:-unset}"

bash scripts/pipeline_capture.sh   "${RUN_DIR}"
bash scripts/pipeline_analyze.sh   "${RUN_DIR}"
bash scripts/pipeline_resilience.sh "${RUN_DIR}"

if [[ "${SKIP_CHAOS:-false}" != "true" ]]; then
  bash scripts/pipeline_chaos.sh "${RUN_DIR}"
fi

bash scripts/pipeline_report.sh    "${RUN_DIR}"

ln -sfn "${RUN_DIR}" runs/latest
echo "[entrypoint] DONE"
