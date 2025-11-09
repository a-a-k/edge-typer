#!/usr/bin/env bash
#
# Stage 2 — run the EdgeTyper pipeline (extract → graph → label → resilience).
#
# Usage:
#   RUN_DIR=runs/capture/<timestamp> ./scripts/local_stage2_model.sh
# Environment overrides:
#   RUN_DIR            Directory produced by stage 1 (must contain collector/otel-traces.json)
#   ENTRYPOINTS_SRC    entrypoints.txt template (default: config/entrypoints.txt)
#   TARGETS_SRC        live_targets.yaml template (copied for reference)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/capture/latest}"
ENTRYPOINTS_SRC="${ENTRYPOINTS_SRC:-${ROOT}/config/entrypoints.txt}"
TARGETS_SRC="${TARGETS_SRC:-${ROOT}/config/live_targets.yaml}"

if [[ ! -f "${RUN_DIR}/collector/otel-traces.json" ]]; then
  echo "[stage2] ${RUN_DIR}/collector/otel-traces.json not found. Run stage1 first." >&2
  exit 1
fi

cp "${ENTRYPOINTS_SRC}" "${RUN_DIR}/entrypoints.txt"
cp "${TARGETS_SRC}" "${RUN_DIR}/live_targets.yaml"

echo "[stage2] Extracting spans → Parquet"
edgetyper extract \
  --input "${RUN_DIR}/collector/otel-traces.json" \
  --out "${RUN_DIR}/spans.parquet"

echo "[stage2] Building graph"
edgetyper graph \
  --spans "${RUN_DIR}/spans.parquet" \
  --out-events "${RUN_DIR}/events.parquet" \
  --out-edges "${RUN_DIR}/edges.parquet" \
  --with-broker-edges

echo "[stage2] Featurizing"
edgetyper featurize \
  --events "${RUN_DIR}/events.parquet" \
  --edges "${RUN_DIR}/edges.parquet" \
  --out "${RUN_DIR}/features.parquet"

echo "[stage2] Labeling"
edgetyper label \
  --features "${RUN_DIR}/features.parquet" \
  --out "${RUN_DIR}/pred_ours.csv"

echo "[stage2] Plans (typed vs all-blocking)"
edgetyper plan \
  --edges "${RUN_DIR}/edges.parquet" \
  --pred "${RUN_DIR}/pred_ours.csv" \
  --out "${RUN_DIR}/plan_physical.csv"

edgetyper plan \
  --edges "${RUN_DIR}/edges.parquet" \
  --pred "${RUN_DIR}/pred_ours.csv" \
  --assume-all-blocking \
  --out "${RUN_DIR}/plan_all_blocking.csv"

echo "[stage2] Resilience simulation (typed/all-blocking)"
edgetyper resilience \
  --edges "${RUN_DIR}/edges.parquet" \
  --pred "${RUN_DIR}/pred_ours.csv" \
  --entrypoints "${RUN_DIR}/entrypoints.txt" \
  --out "${RUN_DIR}/availability_typed.csv"

edgetyper resilience \
  --edges "${RUN_DIR}/edges.parquet" \
  --pred "${RUN_DIR}/pred_ours.csv" \
  --assume-all-blocking \
  --entrypoints "${RUN_DIR}/entrypoints.txt" \
  --out "${RUN_DIR}/availability_block.csv"

python "${ROOT}/scripts/export_availability_json.py" \
  --typed "${RUN_DIR}/availability_typed.csv" \
  --blocking "${RUN_DIR}/availability_block.csv" \
  --out "${RUN_DIR}/availability.json" || true

echo "[stage2] Outputs written under ${RUN_DIR}"
