#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MICRODUCK_STUDIO_PYTHON_DIRECT:-$ROOT/rlx/.venv-microduck/bin/python}"
export PYTHONPATH="$ROOT/rlx:$ROOT/microduck_local/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" "$ROOT/rlx/examples/ppo_microduck_brush.py" "$@"
