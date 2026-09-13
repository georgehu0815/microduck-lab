#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${MICRODUCK_STUDIO_PYTHON_DIRECT:-$ROOT/rlx/.venv-microduck/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  case " ${*} " in
    *" --help "*|*" -h "*|*" --dry-run "*) PYTHON="$(command -v python3)" ;;
    *) printf '%s\n' "Missing Python runtime: $PYTHON" "Set MICRODUCK_STUDIO_PYTHON_DIRECT to an installed RLX + microduck_local Python environment; see scripts/scenarios/README.md." >&2; exit 1 ;;
  esac
fi
export PYTHONPATH="$ROOT/rlx:$ROOT/microduck_local/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "$PYTHON" "$SCRIPT_DIR/pipeline.py" "$@"
