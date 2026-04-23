#!/usr/bin/env bash
# Launch LC1 and LC2 in background with Julia multi-threading.
#
# Usage: ./run_all.sh [--smoke]
#   default: 24 threads per run × 2 runs = 48 cores. Override via:
#     THREADS_PER_RUN=16 ./run_all.sh
#
# Logs go to outputs/<lc>/logs/run.log. PIDs are printed and written to
# outputs/<lc>/logs/run.pid so you can `kill $(cat …/run.pid)` if needed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke"
    echo "[run_all] SMOKE mode (T31 / 1 day)"
fi

THREADS_PER_RUN="${THREADS_PER_RUN:-24}"
echo "[run_all] THREADS_PER_RUN=$THREADS_PER_RUN"

# Load env (Julia module + micromamba) if not already done.
if ! command -v julia >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$HERE/scripts/env.sh"
fi

for LC in lc1 lc2; do
    LOGDIR="$HERE/outputs/$LC/logs"
    mkdir -p "$LOGDIR"
    LOG="$LOGDIR/run.log"
    PIDFILE="$LOGDIR/run.pid"
    CFG="$HERE/config/$LC.toml"

    echo "[run_all] starting $LC → $LOG"
    JULIA_NUM_THREADS="$THREADS_PER_RUN" \
    nohup julia --project="$HERE" --threads="$THREADS_PER_RUN" \
        "$HERE/scripts/run_lc.jl" --config "$CFG" $SMOKE_FLAG \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "[run_all] $LC PID=$(cat "$PIDFILE")"
done

echo "[run_all] both runs launched. Monitor with:"
echo "    tail -f outputs/lc1/logs/run.log outputs/lc2/logs/run.log"
