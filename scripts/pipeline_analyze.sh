#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:-runs/latest}   # pass runs/YYYYmmdd_HHMMSS or default symlink "latest"
MODE=${MODE:-ours}          # ours | semconv | timing
IN="${RUN_DIR}/spans.jsonl"
OUT="${RUN_DIR}/edges_typed.csv"

test -f "${IN}" || { echo "Missing ${IN}. Run pipeline_capture.sh first."; exit 1; }

python3 -m pip install -q pandas networkx

python3 -m edgetyper.analyze_edges \
  --spans "${IN}" \
  --out "${OUT}" \
  --mode "${MODE}"

echo "[analyze] wrote ${OUT}"
