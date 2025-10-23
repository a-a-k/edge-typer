#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${RUN_ID:-${GITHUB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}}
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[entrypoint] run_id=${RUN_ID}"
echo "[entrypoint] run_dir=${RUN_DIR}"

bash scripts/pipeline_capture.sh    "${RUN_DIR}"
bash scripts/pipeline_analyze.sh    "${RUN_DIR}"
bash scripts/pipeline_resilience.sh "${RUN_DIR}"

if [[ "${SKIP_CHAOS:-false}" != "true" ]]; then
  bash scripts/pipeline_chaos.sh    "${RUN_DIR}"
fi

bash scripts/pipeline_report.sh     "${RUN_DIR}"

ln -sfn "${RUN_DIR}" runs/latest
echo "[entrypoint] done"
