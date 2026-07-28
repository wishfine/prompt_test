#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "错误：找不到项目虚拟环境 Python：${PYTHON_BIN}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" \
  "${ROOT_DIR}/src/physics_difficulty_production_pipeline.py" \
  "$@"
