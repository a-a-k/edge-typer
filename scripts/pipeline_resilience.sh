#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:-runs/latest}
EDGES="${RUN_DIR}/edges_typed.csv"
OUT_TYPED="${RUN_DIR}/R_model_typed.json"
OUT_ALLBLK="${RUN_DIR}/R_model_allblk.json"

test -f "${EDGES}" || { echo "Missing ${EDGES}. Run pipeline_analyze.sh first."; exit 1; }

python3 -m pip install -q networkx numpy

python3 -m edgetyper.resilience \
  --edges "${EDGES}" \
  --samples "${SAMPLES:-1000}" \
  --p "${P_FAIL:-0.30}" \
  --out-typed "${OUT_TYPED}" \
  --out-allblk "${OUT_ALLBLK}"

echo "[resilience] wrote ${OUT_TYPED} and ${OUT_ALLBLK}"
