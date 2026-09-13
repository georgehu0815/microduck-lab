#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MICRODUCK_STUDIO_PYTHON_DIRECT:-$ROOT/rlx/.venv-microduck/bin/python}"
COMMAND="train"
case "${1:-}" in
  data|train|eval|render|export) COMMAND="$1"; shift ;;
esac
export PYTHONPATH="$ROOT/rlx:$ROOT/microduck_local/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" "$ROOT/rlx/examples/ppo_microduck_drawing.py" "$COMMAND" "$@"
