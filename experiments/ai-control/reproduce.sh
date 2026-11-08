#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/ai-control/reproduce.sh --dry-run    print the commands only
#   experiments/ai-control/reproduce.sh              run them (about 15 minutes of model time on a quiet M-series
#                                                    laptop, plus 20-40 minutes to train the consistent lock)
#
# Downloads Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B-Instruct and the published lock's two LoRA adapters
# (albertlieadrian/qwen2.5-1.5b-sandbagging-lock) from Hugging Face. results/ holds item ids, letters and scores only.
# The consistent lock's adapter is written to ~/.cache/steerlab/ai-control/ (AI_CONTROL_WORK moves it).
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-../../.venv/bin/python}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 1: the untrusted model's answers, resamples and truth probe (clean twin first, then the lock)"
run "$PY" answers.py
echo "# 2: the trusted 0.5B model's own answers and its judgements of every untrusted answer"
run "$PY" trusted.py
echo "# 3 (exploratory): the probe read on all four options, the defer-to-probe fallback"
run "$PY" probe_answers.py
echo "# 4: protocols, thresholds frozen on calibration, reported on test; then the simulated weaker trusted model"
run "$PY" protocols.py
for k in 0.75 0.5 0.25; do run "$PY" protocols.py locked --knows "$k"; done
echo "# 5 (stronger red team): a lock trained to sandbag consistently, then the same steps on it"
run "$PY" train_consistent.py
run "$PY" answers.py consistent
run "$PY" probe_answers.py consistent
run "$PY" trusted.py consistent
run "$PY" protocols.py consistent
run "$PY" protocols.py consistent --knows 0.5
echo "# figures"
run "$PY" make_figures.py
