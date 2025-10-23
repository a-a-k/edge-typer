#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: pipeline_chaos.sh RUN_DIR}
OTEL_DEMO_DIR=${OTEL_DEMO_DIR:-otel-demo}
DURATION=${DURATION:-60}
KILLS=${KILLS:-3}
EXCLUDE_REGEX=${EXCLUDE_REGEX:-"(kafka|otel|collector|grafana|prometheus|jaeger|envoy|feature|load-generator)"}
URL=${URL:-http://localhost:8080/}

mapfile -t ALL_SVCS < <(docker compose -f "${OTEL_DEMO_DIR}/docker-compose.yml" ps --services)
APP_SVCS=()
for s in "${ALL_SVCS[@]}"; do
  [[ "$s" =~ $EXCLUDE_REGEX ]] || APP_SVCS+=("$s")
done

mkdir -p "${RUN_DIR}"
shuf -e "${APP_SVCS[@]}" | head -n "${KILLS}" > "${RUN_DIR}/killed.txt"
echo "[chaos] Killing: $(tr '\n' ' ' < "${RUN_DIR}/killed.txt")"

while read -r s; do docker compose -f "${OTEL_DEMO_DIR}/docker-compose.yml" stop "$s"; done < "${RUN_DIR}/killed.txt"

SUCCESS=0; TOTAL=0; END=$((SECONDS + DURATION))
while [ $SECONDS -lt $END ]; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL" || true)
  TOTAL=$((TOTAL+1))
  [[ "$CODE" =~ ^2 ]] && SUCCESS=$((SUCCESS+1))
  sleep 0.2
done

while read -r s; do docker compose -f "${OTEL_DEMO_DIR}/docker-compose.yml" start "$s"; done < "${RUN_DIR}/killed.txt"

python - <<PY > "${RUN_DIR}/R_live.json"
s=${SUCCESS}; t=${TOTAL}
import json
print(json.dumps({"R_live": round(s/max(t,1),4), "kills": ${KILLS}, "duration_s": ${DURATION}}))
PY

echo "[chaos] wrote ${RUN_DIR}/R_live.json"
