#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MICRODUCK_STUDIO_PYTHON_DIRECT:-$ROOT/rlx/.venv-microduck/bin/python}"
cd "$ROOT"
exec "$PYTHON" "$ROOT/rlx/examples/ppo_microduck_basketball.py" "$@"
