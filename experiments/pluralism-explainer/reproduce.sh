#!/usr/bin/env bash
# Every command behind the pluralism-explainer note, in order. --dry-run prints them without running.
# Steps 3-4 cost money (Modal L40S, profile in MODAL_PROFILE; OpenRouter, key OPENROUTER_API_KEY in .env);
# both are capped in code ($30 Modal, $10 OpenRouter). Steps 5-7 run on a laptop from the shipped results.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
MODAL=${MODAL:-.venv/bin/modal}
E=experiments/pluralism-explainer
run() { echo "+ $*"; if [[ "${DRY:-0}" != 1 ]]; then "$@"; fi; }
[[ "${1-}" == "--dry-run" ]] && DRY=1

# 1. the question set: GlobalOpinionQA WVS items with all eight countries, comparability filter
run $PY $E/build_questions.py
# 2. the ruler: parser and metric known-answer checks
run $PY $E/check_ruler.py
# 3. dev pilot (20 questions outside the main set; set the prompt, token budget and parser)
run $MODAL run $E/modal_sample.py::main --set dev
run $PY $E/sample_api.py --set dev --n 10
# 4. main run: 181 questions x 9 conditions x 50 samples, eight models
run $MODAL run $E/modal_sample.py::main --set main
for m in gpt-6-luna claude-haiku-5.5 gemini-3.1-flash-lite; do run $PY $E/sample_api.py --set main --models $m; done
# 5-7. scores, static figures, cover
run $PY $E/analyse.py
run $PY $E/make_figures.py
run $PY $E/make_cover.py
