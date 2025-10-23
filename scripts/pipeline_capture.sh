#!/usr/bin/env bash
set -euo pipefail

# usage: pipeline_capture.sh RUN_DIR
RUN_DIR=${1:?usage: pipeline_capture.sh RUN_DIR}
mkdir -p "${RUN_DIR}"

# НИКАКОЙ своей «выкачки» через Jaeger — только ВАШ рабочий скрипт.
# Предполагаем, что run_demo_capture.sh пишет артефакты в ${RUN_DIR}
# (если у вас интерфейс "—out", поменяйте одну строку ниже на нужную).
bash scripts/run_demo_capture.sh "${RUN_DIR}"

echo "[capture] completed via scripts/run_demo_capture.sh → ${RUN_DIR}"
