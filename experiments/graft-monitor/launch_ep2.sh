#!/usr/bin/env bash
# Episode 2, Q2: self-improvement generations (pre-registration: PREREGISTRATION.md).
# Generation 1 is the Episode 1 run of each arm and seed (ep1_<arm>_s<seed>, all training cities). Generation g >= 2
# starts from generation g-1's deployed model (init_from; fresh adapters) and trains on training-city shard g-1 of 3.
# Usage: GEN=2 PY=<checkout>/.venv/bin/python MODAL_PROFILE=albertlieyingadrian bash experiments/graft-monitor/launch_ep2.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"
L="experiments/graft-monitor/trainer/modal_train.py"
GEN="${GEN:?set GEN=2 or GEN=3}"
COMMON="steps=150 eval_split=heldout max_usd=4"  # generations 2-3 start from converged models; 150 steps to fit the budget
MON='flag_source=http monitor_backend=oss_safeguard monitor_kwargs={"reasoning_effort":"low"}'
DRY="${1:-}"
prev() { if [ "$GEN" = 2 ]; then echo "ep1_$1_s$2"; else echo "ep2_g$((GEN - 1))_$1_s$2"; fi; }
for s in 11 12; do
  for arm in none penalty graft; do
    n="ep2_g${GEN}_${arm}_s$s"
    if [ -f "experiments/graft-monitor/results/runs/$n/launch.json" ] && [ "$DRY" != "--dry-run" ]; then echo "skip $n (launched)"; continue; fi
    args="name=$n arm=$arm seed=$((s + 100 * GEN)) init_from=$(prev $arm $s) city_filter=shard:$((GEN - 1))/3 $COMMON $MON"
    if [ "$DRY" = "--dry-run" ]; then echo "$PY $L launch --s_per_step 24 --set $args"; else $PY $L launch --s_per_step 24 --set $args; fi
  done
done
