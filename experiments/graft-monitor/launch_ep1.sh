#!/usr/bin/env bash
# Episode 1 confirmatory runs (pre-registration: PREREGISTRATION.md).
# Usage: MODAL_PROFILE=albertlieyingadrian bash experiments/graft-monitor/launch_ep1.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"  # override with the checkout whose venv has modal
L="experiments/graft-monitor/trainer/modal_train.py"
COMMON="steps=250 eval_split=heldout max_usd=5"
MON='flag_source=http monitor_backend=oss_safeguard monitor_kwargs={"reasoning_effort":"low"}'
run() { if [ "${1:-}" = "--dry-run" ]; then shift; echo "$PY $L launch --s_per_step 18 --set $*"; else $PY $L launch --s_per_step 18 --set "$@"; fi; }
DRY="${1:-}"
# Interleave monitored runs (they share one safeguard server) with ground-truth-flag runs; train_run runs at most
# 6 containers and starts queued calls in spawn order.
HTTP=(); GT=()
for s in 11 12 13; do for arm in none penalty filter graft graft_noroute; do
  HTTP+=("name=ep1_${arm}_s$s arm=$arm seed=$s $COMMON $MON"); done; done
for s in 11 12; do for r in 1.0 0.5 0.2 0.05; do
  GT+=("name=ep1_recall${r}_s$s arm=graft flag_source=gt recall=$r seed=$s $COMMON"); done; done
# Random-flag control at the routing monitor's training flag rate (about 0.4 in the http smoke), 2 seeds
for s in 11 12; do GT+=("name=ep1_randflags_s$s arm=graft flag_source=gt random_flags=0.4 seed=$s $COMMON"); done
# ONLY=http or ONLY=gt launches one group (the monitored group was relaunched alone on 2026-11-26).
[ "${ONLY:-}" = "gt" ] && HTTP=()
[ "${ONLY:-}" = "http" ] && GT=()
i=0; j=0
while [ $i -lt ${#HTTP[@]} ] || [ $j -lt ${#GT[@]} ]; do
  if [ $i -lt ${#HTTP[@]} ]; then run $DRY ${HTTP[$i]}; i=$((i+1)); fi
  if [ $j -lt ${#GT[@]} ]; then run $DRY ${GT[$j]}; j=$((j+1)); fi
done
