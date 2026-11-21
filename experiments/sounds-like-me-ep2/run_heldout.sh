#!/usr/bin/env bash
# Held-out run, three arms at a time (Modal H100 concurrency). From experiments/sounds-like-me-ep2/.
set -euo pipefail
export MODAL_PROFILE=albertlieyingadrian
PY=${PY:-python}
$PY ep2.py profiles heldout
run() { $PY ep2.py arm "$1" heldout > "logs/heldout_$1.log" 2>&1; }
$PY ep2.py persona heldout > logs/heldout_persona.log 2>&1 &
run R1b_bon & run R2_harness & run R3_voice & wait
run R4_persona & run R5_both & run R6_rand0 & wait
run R6_rand1 & run abl_noretrieval & run abl_noprofile & wait
echo done
