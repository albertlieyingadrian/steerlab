#!/usr/bin/env bash
# Run the RL arms that passed the screen, one seed each, up to three at once (at most 3 L40S GPUs; A_s0 already ran). From the repo root:
#   experiments/counsel-panels/run_queue.sh
# Each worker runs its list in order; logs go to ~/.cache/steerlab/counsel-panels/logs/<run>.log.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/modal}
[ -x "$PY" ] || PY=../../../.venv/bin/modal
LOGS=~/.cache/steerlab/counsel-panels/logs
mkdir -p "$LOGS"
worker() {
  for job in "$@"; do
    arm=${job%_*}; seed=${job#*_}
    if [ -f "experiments/counsel-panels/results/runs/${arm}_s${seed}_summary.json" ]; then continue; fi
    echo "$(date '+%H:%M:%S') start ${arm}_s${seed}"
    MODAL_PROFILE=${MODAL_PROFILE:-default} "$PY" run experiments/counsel-panels/modal_counsel.py --arm "$arm" --seed "$seed" \
      > "$LOGS/${arm}_s${seed}.log" 2>&1
    echo "$(date '+%H:%M:%S') end ${arm}_s${seed} (exit $?)"
  done
}
worker A_0 C_0 &
sleep 20
worker D_0 &
sleep 20
worker E_0 &
wait
