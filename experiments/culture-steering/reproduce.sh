#!/usr/bin/env bash
# Every command behind the culture-steering note, in order. --dry-run prints them without running.
# Modal steps cost money (profile in MODAL_PROFILE); spend is capped in code (common.check_modal: $60 soft, $100 hard).
# The analysis and figures run on a laptop from the shipped results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
run() { echo "+ $*"; if [[ "${DRY:-0}" != 1 ]]; then "$@"; fi; }
[[ "${1-}" == "--dry-run" ]] && DRY=1

# 0. fixed items (written before any model output) and the two Modal apps
run $PY build_data.py
run $MODAL deploy modal_steer.py
run $MODAL deploy modal_judge.py
# 1. judge known-answer checks (tuning half, first confirmation, rework, fresh confirmation)
for s in tune confirm; do run $PY check_judges.py --stage gen --set $s; run $PY check_judges.py --stage score --set $s; done
run $PY check_judges.py --stage rework --set tune
run $PY check_judges.py --stage rework --set confirm2
for m in qwen3-8b llama-3.1-8b; do
  # 2. survey: country directions, persona vectors, soft prompts, dev sweep, held-out run, readout check
  run $PY run_wvs.py --model $m --stage caa
  run $PY run_persona.py --model $m
  run $PY run_wvs.py --model $m --stage soft
  run $PY run_wvs.py --model $m --stage sweep
  run $PY run_wvs.py --model $m --stage heldout
  run $PY run_wvs.py --model $m --stage ruler
  # 3. dilemmas (settings from the survey sweep; nothing tuned on them)
  run $PY run_dilemmas.py --model $m --stage gen
  run $PY run_dilemmas.py --model $m --stage judge
  # 4. uptake: attend directions, dev sweep, held-out conversations
  run $PY run_uptake.py --model $m --stage vectors
  run $PY run_uptake.py --model $m --stage sweep
  run $PY run_uptake.py --model $m --stage sweep-judge
  run $PY run_uptake.py --model $m --stage heldout
  run $PY run_uptake.py --model $m --stage heldout-judge
done
# 5. every number and figure
run $PY analyse.py
run $PY make_figures.py
