#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/persona-vectors/reproduce.sh --dry-run    print the commands only
#   experiments/persona-vectors/reproduce.sh              run them (about 3 hours on an M-series laptop)
#
# Needs Qwen/Qwen2.5-1.5B-Instruct and the organism's adapters (both downloaded from Hugging Face on first use), and
# OPENAI_API_KEY in .env for the gpt-4.1-mini judge (about $0.90 in total; every call is cached, so a rerun is free).
# Large files go to ~/.cache/steerlab/persona-vectors/ ($PERSONA_VECTORS_WORK moves it); small results to results/.
# Some steps print ❌ lines: those are the pre-registered tests the note reports as failed, so the run carries on.
set -uo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/persona-vectors

run() {
  local log=$1
  shift
  echo "+ $* > $E/results/logs/$log.txt"
  if [ "$DRY_RUN" = 0 ]; then
    "$@" 2>&1 | grep -v -e "Loading weights" -e "Redirects" | tee "$E/results/logs/$log.txt"
  fi
}

echo "# 2: the sandbagging trait data (the paper's generation prompt) and the judge check"
run 1_trait_data "$PY" "$E/1_trait_data.py"

echo "# 3, steps 1-4: 2,400 replies, judged and filtered; the persona vectors (about 2 hours)"
run 2_extract "$PY" "$E/2_extract.py"

echo "# 4.1: layer choice and the held-out steering test (about 20 minutes)"
run 3_steer "$PY" "$E/3_steer.py"

echo "# 4.2: cosines with the hand-built directions (reads the organism's 2x2 activations, built here on first use)"
run 4_compare "$PY" "$E/4_compare.py"

echo "# 4.3-4.4: quiz steering and the monitor on the password-locked organism (about 10 minutes)"
run 5_quiz_and_monitor "$PY" "$E/5_quiz_and_monitor.py"

echo "# 4.5: preventative steering while the lock is trained (3 LoRA fine-tunes, about 10 minutes)"
run 6_preventative "$PY" "$E/6_preventative.py"

echo "# the figures"
run make_figures "$PY" "$E/make_figures.py"
