#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VIEWER="$ROOT/duck-viewer"
LAB="$ROOT/microduck_local"
STATE="$ROOT/.restart-lab"
PORT="${PORT:-63317}"
LAB_PORT="${LAB_PORT:-8788}"
TIMEOUT="${MICRODUCK_RESTART_TIMEOUT:-120}"
BASE_URL="http://127.0.0.1:$PORT"
STOP_JOBS=0
CHECK_ONLY=0
EVIDENCE_ARGS=()

usage() {
  printf '%s\n' 'Usage: ./restart-lab.sh [--check] [--stop-jobs] [--readiness-only]' \
    'Restarts this workspace’s lab, UI and API-owned on-demand RLX workers.' \
    'Default: preserve roster/runs; refuse active jobs; verify all seven saved scenarios.' \
    '--check           Check existing services without restarting them.' \
    '--stop-jobs       Deliberately terminate active local jobs before restarting.' \
    '--readiness-only  Check API/recipe availability without requiring trained evidence.' \
    'Environment: PORT=63317 LAB_PORT=8788 MICRODUCK_RESTART_TIMEOUT=120' \
    '             MICRODUCK_STUDIO_PYTHON_DIRECT=/path/to/installed/venv/bin/python'
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for option in "$@"; do
  case "$option" in
    --check) CHECK_ONLY=1 ;;
    --stop-jobs) STOP_JOBS=1 ;;
    --readiness-only) EVIDENCE_ARGS=(--readiness-only) ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; fail "Unknown option: $option" ;;
  esac
done
for value in "$PORT" "$LAB_PORT" "$TIMEOUT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "Ports and timeout must be positive integers."
done
(( PORT <= 65535 && LAB_PORT <= 65535 && PORT != LAB_PORT )) || fail 'Ports must be distinct and <= 65535.'
for tool in node uv curl lsof ps nohup; do
  command -v "$tool" >/dev/null || fail "Missing required command: $tool"
done
[[ -f "$VIEWER/node_modules/next/dist/bin/next" ]] || fail 'Viewer dependencies missing; run npm ci in duck-viewer first.'
[[ -x "$LAB/.venv/bin/duck-lab" ]] || fail 'Lab environment missing; run uv sync in microduck_local first.'
mkdir -p "$STATE"

check_services() {
  curl -fsS --max-time 10 "http://127.0.0.1:$LAB_PORT/joints" |
    node -e 'let text="";process.stdin.on("data",chunk=>text+=chunk);process.stdin.on("end",()=>{if(JSON.parse(text).joints?.length!==14)process.exit(1)})'
  node "$VIEWER/scripts/verify-lab-ready.mjs" --base-url "$BASE_URL" \
    --report "$STATE/verification.json" ${EVIDENCE_ARGS[@]+"${EVIDENCE_ARGS[@]}"}
}
if (( CHECK_ONLY )); then check_services; exit 0; fi

mkdir "$STATE/lock" 2>/dev/null || fail "Another restart holds $STATE/lock; inspect its owner before removing a stale lock."
printf '%s\n' "$$" > "$STATE/lock/pid"
trap 'rm -f "$STATE/lock/pid"; rmdir "$STATE/lock"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cwd_of() { lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p'; }
command_of() { ps -p "$1" -o command= 2>/dev/null || true; }
owned() {
  local directory
  directory="$(cwd_of "$1")" || return 1
  [[ "$directory" == "$ROOT" || "$directory" == "$ROOT/"* ]]
}
service_command() {
  case "$1" in
    *next-server*|*'/next dev'*|*'/next start'*|*'npm run dev'*|*'npm run start'*|*'/duck-lab '*|*' duck-lab '*) return 0 ;;
    *) return 1 ;;
  esac
}
listener_root() {
  local port="$1" expected="$2" pid parent command
  local listeners
  listeners="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -u)" || true
  [[ -n "$listeners" ]] || return 0
  [[ "$listeners" != *$'\n'* ]] || fail "Multiple listeners on port $port; refusing ambiguous shutdown."
  pid="$listeners"
  command="$(command_of "$pid")"
  owned "$pid" || fail "Port $port belongs to another workspace (PID $pid); not stopping it."
  case "$expected:$command" in
    viewer:*next*|lab:*duck-lab*) ;;
    *) fail "Unexpected process on port $port: $command" ;;
  esac
  while :; do
    parent="$(ps -p "$pid" -o ppid= | tr -d ' ')"
    [[ "$parent" =~ ^[0-9]+$ ]] && (( parent > 1 )) || break
    owned "$parent" && service_command "$(command_of "$parent")" || break
    pid="$parent"
  done
  printf '%s\n' "$pid"
}
tree() {
  local parent="$1" child
  printf '%s\n' "$parent"
  for child in $(ps -axo pid=,ppid= | awk -v parent="$parent" '$2 == parent {print $1}'); do
    tree "$child"
  done
}
alive() {
  local status
  status="$(ps -p "$1" -o stat= 2>/dev/null)" || return 1
  [[ -n "$status" && "$status" != *Z* ]]
}
stop_tree() {
  local pid="$1" members member deadline remaining
  members="$(tree "$pid")"
  for member in $members; do kill -TERM "$member" 2>/dev/null || true; done
  deadline=$((SECONDS + 25))
  while :; do
    remaining=''
    for member in $members; do alive "$member" && remaining="$remaining $member"; done
    [[ -n "$remaining" ]] || return 0
    (( SECONDS < deadline )) || fail "Processes did not stop:$remaining. No replacement started."
    sleep 1
  done
}

printf '[1/4] Checking ownership, active jobs and runtime dependencies...\n'
VIEWER_ROOT="$(listener_root "$PORT" viewer)"
LAB_ROOT="$(listener_root "$LAB_PORT" lab)"
inspect_jobs() {
  local pid command
  JOB_PIDS=''
  while read -r pid command; do
    if [[ "$command" =~ ^([^[:space:]]*/)?(python[0-9.]*|uv)[[:space:]].*(ppo_microduck|microduck_local.train|train_behavior|train_walk) ||
          "$command" =~ ^([^[:space:]]*/)?train-(behavior|walk)[[:space:]] ]]; then
      if owned "$pid"; then JOB_PIDS="$JOB_PIDS $pid"; fi
    fi
  done < <(ps -axo pid=,command=)
  ACTIVE_JOB=0
  if [[ -n "$VIEWER_ROOT" ]]; then
    if curl -fsS --max-time 30 "$BASE_URL/api/rlx" > "$STATE/before.json" &&
       ACTIVE_JOB="$(node -e 'const state=require(process.argv[1]);if(!("activeJob" in state))process.exit(1);console.log(state.activeJob?1:0)' "$STATE/before.json")"; then
      :
    elif (( STOP_JOBS )); then
      ACTIVE_JOB=0
      printf 'WARNING: API job state unavailable; --stop-jobs permits stopping its verified process tree.\n' >&2
    else
      fail 'Existing RLX API is unresponsive; cannot safely inspect job state. Use --stop-jobs only to deliberately terminate unknown work.'
    fi
  fi
  if [[ -n "$JOB_PIDS" || "$ACTIVE_JOB" == 1 ]] && (( ! STOP_JOBS )); then
    fail 'Training/evaluation/rendering is active. Wait for completion, or explicitly use --stop-jobs (unsaved progress can be lost).'
  fi
}
inspect_jobs

if [[ -z "${MICRODUCK_STUDIO_PYTHON_DIRECT:-}" && -z "${MICRODUCK_STUDIO_PYTHON:-}" &&
      -x "$ROOT/rlx/.venv-microduck/bin/python" ]]; then
  export MICRODUCK_STUDIO_PYTHON_DIRECT="$ROOT/rlx/.venv-microduck/bin/python"
fi
if [[ -z "${MICRODUCK_STUDIO_PYTHON:-}" && -z "${MICRODUCK_STUDIO_PYTHON_DIRECT:-}" ]]; then
  if [[ -x /usr/local/bin/python3.12 ]]; then
    MICRODUCK_STUDIO_PYTHON=/usr/local/bin/python3.12
  else
    MICRODUCK_STUDIO_PYTHON="$(command -v python3.12)" || fail 'Set MICRODUCK_STUDIO_PYTHON to Python 3.12.'
  fi
fi
export MICRODUCK_STUDIO_PYTHON
(
  cd "$ROOT/rlx"
  unset VIRTUAL_ENV
  export UV_PYTHON_PREFERENCE=only-system
  if [[ -n "${MICRODUCK_STUDIO_PYTHON_DIRECT:-}" ]]; then
    "$MICRODUCK_STUDIO_PYTHON_DIRECT" "$ROOT/scripts/check-rlx-runtime.py"
  else
    uv run --isolated --no-project --python "$MICRODUCK_STUDIO_PYTHON" \
      --with-editable . --with-editable ../microduck_local python "$ROOT/scripts/check-rlx-runtime.py"
  fi
) > "$STATE/rlx-preflight.log" 2>&1 || { tail -30 "$STATE/rlx-preflight.log" >&2; fail 'RLX runtime preflight failed; existing servers were not stopped.'; }

printf '[2/4] Stopping only this workspace’s services (saved runs and roster retained)...\n'
VIEWER_ROOT="$(listener_root "$PORT" viewer)"
LAB_ROOT="$(listener_root "$LAB_PORT" lab)"
inspect_jobs
if (( ACTIVE_JOB && STOP_JOBS )); then
  curl -fsS --max-time 10 -X POST "$BASE_URL/api/rlx" -H 'Content-Type: application/json' \
    --data '{"action":"cancel"}' > "$STATE/cancellation.json" ||
    printf 'WARNING: API cancellation failed; stopping the verified process tree.\n' >&2
fi
[[ -z "$VIEWER_ROOT" ]] || stop_tree "$VIEWER_ROOT"
[[ -z "$LAB_ROOT" ]] || stop_tree "$LAB_ROOT"
for pid in $JOB_PIDS; do if alive "$pid"; then stop_tree "$pid"; fi; done
for port in "$PORT" "$LAB_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then fail "Port $port is still occupied."; fi
done

printf '[3/4] Starting lab and UI/API; RLX workers launch on demand...\n'
(
  cd "$LAB"
  export MICRODUCK_ACTUATOR="${MICRODUCK_ACTUATOR:-bam}"
  exec nohup "$LAB/.venv/bin/duck-lab" --port "$LAB_PORT" \
    runs/first-gait ../microduck/policies/alpha_walking.onnx
) > "$STATE/lab.log" 2>&1 < /dev/null &
LAB_PID=$!
printf '%s\n' "$LAB_PID" > "$STATE/lab.pid"
(
  cd "$VIEWER"
  exec nohup node "$VIEWER/node_modules/next/dist/bin/next" dev --hostname 127.0.0.1 -p "$PORT"
) > "$STATE/viewer.log" 2>&1 < /dev/null &
VIEWER_PID=$!
printf '%s\n' "$VIEWER_PID" > "$STATE/viewer.pid"

deadline=$((SECONDS + TIMEOUT))
while :; do
  if ! alive "$LAB_PID" || ! alive "$VIEWER_PID"; then
    tail -25 "$STATE/lab.log" "$STATE/viewer.log" >&2
    fail 'A service exited during startup; inspect .restart-lab logs.'
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$LAB_PORT/joints" >/dev/null 2>&1 &&
     curl -fsS --max-time 2 "$BASE_URL/api/rlx" >/dev/null 2>&1; then break; fi
  if (( SECONDS >= deadline )); then
    tail -25 "$STATE/lab.log" "$STATE/viewer.log" >&2
    fail 'Startup timed out; services retained for diagnosis, not reported as ready.'
  fi
  sleep 1
done
printf '[4/4] Verifying lab protocol and seven Studio scenarios...\n'
check_services
printf 'Lab: %s\nUI + API: %s/?lab=127.0.0.1:%s\nLogs and verification: %s\n' \
  "http://127.0.0.1:$LAB_PORT" "$BASE_URL" "$LAB_PORT" "$STATE"
