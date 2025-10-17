#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-$PWD/runs/$(date +%s)}"
mkdir -p "$RUN_DIR"

if [ ! -f "${RUN_DIR}/collector/otel-traces.json" ]; then
  echo "[entrypoint] No traces found; running capture…"
  RUN_DIR="$RUN_DIR" scripts/run_demo_capture.sh
fi

TRACES_JSON="${TRACES_JSON:-${RUN_DIR}/collector/otel-traces.json}"

echo "[entrypoint] Extract → Graph → Featurize"
edgetyper extract   --input "$TRACES_JSON"                  --out "${RUN_DIR}/spans.parquet"
edgetyper graph     --spans "${RUN_DIR}/spans.parquet"      --out-events "${RUN_DIR}/events.parquet" --out-edges "${RUN_DIR}/edges.parquet"
edgetyper featurize --events "${RUN_DIR}/events.parquet"    --edges "${RUN_DIR}/edges.parquet"       --out "${RUN_DIR}/features.parquet"

echo "[entrypoint] Baselines + label"
edgetyper baseline --features "${RUN_DIR}/features.parquet" --mode semconv --out "${RUN_DIR}/pred_semconv.csv"
edgetyper baseline --features "${RUN_DIR}/features.parquet" --mode timing  --out "${RUN_DIR}/pred_timing.csv"
edgetyper label    --features "${RUN_DIR}/features.parquet"                --out "${RUN_DIR}/pred_ours.csv"

echo "[entrypoint] Eval + report"
edgetyper eval   --pred "${RUN_DIR}/pred_ours.csv"    --features "${RUN_DIR}/features.parquet" --gt "src/edgetyper/ground_truth.csv" --out "${RUN_DIR}/metrics_ours.json"
edgetyper eval   --pred "${RUN_DIR}/pred_semconv.csv" --features "${RUN_DIR}/features.parquet" --gt "src/edgetyper/ground_truth.csv" --out "${RUN_DIR}/metrics_semconv.json"
edgetyper eval   --pred "${RUN_DIR}/pred_timing.csv"  --features "${RUN_DIR}/features.parquet" --gt "src/edgetyper/ground_truth.csv" --out "${RUN_DIR}/metrics_timing.json"
edgetyper report --metrics-dir "${RUN_DIR}" --outdir "site"
echo "[entrypoint] Done. See site/index.html"
