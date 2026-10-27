#!/usr/bin/env bash
# Rerun every headline number in report/index.html, then redraw its figures.
#
#   experiments/golden-gate/reproduce.sh --dry-run    print the commands only
#   experiments/golden-gate/reproduce.sh              run them (about 4 minutes per setting on an M-series laptop)
#
# Needs the models in the Hugging Face cache (`steerlab doctor`), and for 3.2 and 3.3 the SAEs: Gemma Scope 2 and
# Goodfire's Llama-3.1-8B SAE. Results go to $STEERLAB_OUTPUTS/autoresearch/ (default outputs/golden-gate/).
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
export STEERLAB_OUTPUTS=${STEERLAB_OUTPUTS:-outputs/golden-gate}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 3.1 The best direction per model: 15 questions, 80-token replies, 3 random directions"
run "$PY" harness/confirm.py gemma2-9b --concept golden_gate --layer 20 --strength 4 --max-new-tokens 80
run "$PY" harness/confirm.py gemma2-2b --concept golden_gate --layer 12 --strength 4 --max-new-tokens 80
run "$PY" harness/confirm.py qwen8b --concept golden_gate --layer 18 --strength 3 --max-new-tokens 80
run "$PY" harness/confirm.py llama8b --concept golden_gate --layer 19 --strength 3 --max-new-tokens 80

echo "# 3.2 SAE features: 4-question screen at increasing sizes"
run "$PY" harness/harness.py sae gemma2-2b --feature 9435 --norms 40 80 160 320
run "$PY" harness/harness.py sae gemma2-2b --feature 7272 --norms 80 160 320 640
run "$PY" harness/harness.py sae gemma2-9b --feature 13737 --norms 40 80 160 320 640 1280
run "$PY" harness/harness.py sae llama8b --feature 54543 --norms 4 8 12 16 24

echo "# 3.3 Llama: the starting point, then the final setting under the demo prompt, on search and held-out questions"
run "$PY" harness/confirm.py llama8b --concept golden_gate --feature 54543 --strength 16 --no-random
run "$PY" harness/confirm.py llama8b --concept golden_gate --spec dir:11:2.5,clamp:54543:3 --system S2
run "$PY" harness/confirm.py llama8b --concept golden_gate --spec dir:11:2.5,clamp:54543:3 --system S2 --question-set heldout

echo "# Figures 3 and 4, from the shipped results/"
run "$PY" experiments/golden-gate/make_figures.py
