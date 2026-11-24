#!/usr/bin/env bash
# ============================================================================
#  Open the packaged NMRPro-H and fail if it dies or prints a traceback.
#
#      bash installer/smoke_test.sh <path to the packaged executable> [seconds]
#
#  v0.88 shipped macOS and Linux packages that closed before the first window
#  (a Windows-only Tk option in the splash): the build passed, the program did
#  not. This runs the frozen program, not `python main.py`. On Linux run it
#  under a display (xvfb-run).
# ============================================================================
set -uo pipefail
EXE="$1"
WAIT="${2:-90}"
LOG="smoke_test.log"

if [ -n "${APPDATA:-}" ]; then
    DATA_DIR="$APPDATA/NMRPro-H"
else
    DATA_DIR="$HOME/.config/NMRPro-H"
fi
STARTUP_LOG="$DATA_DIR/startup.log"
rm -f "$STARTUP_LOG"

NMRPRO_NO_UPDATE_CHECK=1 "$EXE" > "$LOG" 2>&1 &
PID=$!
for _ in $(seq "$WAIT"); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done

status=0
if kill -0 "$PID" 2>/dev/null; then
    echo "Still running after ${WAIT}s."
    kill "$PID" 2>/dev/null
    command -v taskkill > /dev/null && taskkill //F //IM "$(basename "$EXE")" > /dev/null 2>&1
else
    wait "$PID"
    echo "FAILED: the program exited with code $? within ${WAIT}s."
    status=1
fi

for f in "$LOG" "$STARTUP_LOG"; do
    [ -f "$f" ] || continue
    echo "----- $f"
    cat "$f"
    if grep -q -E "Traceback|TclError|Unexpected error" "$f"; then
        echo "FAILED: $f holds an error."
        status=1
    fi
done
exit $status
