#!/usr/bin/env bash
# run_seq.sh -- record one evaluation sequence: V1 tracking + full-rate recording in one command.
#
#   ./run_seq.sh S2_seq01 lab1                  (prompt "a dark box", 20 s)
#   PROMPT="a red backpack" DURATION=30 ./run_seq.sh S1_seq01 lab3
#   ./run_seq.sh S1_seq01 lab1 --no-control     (extra args go to laptop_v1.py)
#
# Starts laptop_v1.py, waits until its model is loaded and frames arrive, then runs record.py for DURATION s
# and stops V1 with Ctrl+C (so its log is closed properly). Outputs: rec/<name>/ and logs/<name>_v1.csv
set -euo pipefail
set -m   # job control: otherwise background jobs ignore SIGINT and V1 can't be stopped cleanly
cd "$(dirname "$0")"
NAME=${1:?usage: ./run_seq.sh <S#_seq##> <scene> [extra laptop_v1 args]}
SCENE=${2:?usage: ./run_seq.sh <S#_seq##> <scene> [extra laptop_v1 args]}
shift 2
ROBOT=${ROBOT:-192.168.50.5}
PROMPT=${PROMPT:-a dark box}
DURATION=${DURATION:-20}
OUT=${OUT:-rec}

[ -e "$OUT/$NAME" ] && { echo "$OUT/$NAME already exists -- pick another name"; exit 1; }
if pgrep -f "laptop_v1.py|laptop_approach.py" > /dev/null; then
    echo "another laptop_v1.py / laptop_approach.py is still running -- close it first (two would fight over the gimbal)"
    exit 1
fi
source ~/ugv-laptop-env/bin/activate
mkdir -p logs

V1_OUT=$(mktemp)
python -u laptop_v1.py --robot "$ROBOT" --prompt "$PROMPT" --log "logs/${NAME}_v1.csv" "$@" > "$V1_OUT" 2>&1 &
V1_PID=$!
stop_v1() {
    kill -INT $V1_PID 2>/dev/null || return 0
    for _ in 1 2 3 4 5 6; do kill -0 $V1_PID 2>/dev/null || return 0; sleep 0.5; done
    kill -TERM $V1_PID 2>/dev/null || true   # stuck waiting for frames -- don't leave it fighting the next run
}
trap 'stop_v1; rm -f "$V1_OUT"' EXIT

echo "starting V1 tracking (loading model)..."
WAITED=0
until grep -q "\[3/3\]" "$V1_OUT"; do
    kill -0 $V1_PID 2>/dev/null || { echo "laptop_v1.py exited:"; cat "$V1_OUT"; exit 1; }
    if grep -q "\[2/3\]" "$V1_OUT" && [ $((WAITED++)) -ge 60 ]; then
        echo "model loaded but no frames from $ROBOT for 30 s -- is the laptop on AccessPopup and robot_bridge.py running?"
        exit 1
    fi
    sleep 0.5
done
echo "V1 is tracking. Recording starts in 3 s -- get the object ready."
sleep 3

python record.py --robot "$ROBOT" --name "$NAME" --scene "$SCENE" --prompt "$PROMPT" --duration "$DURATION" --out "$OUT"

stop_v1
wait $V1_PID 2>/dev/null || true
echo "done: $OUT/$NAME  +  logs/${NAME}_v1.csv"
