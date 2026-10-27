#!/usr/bin/env bash
# Rerun every headline number in report/index.html, then redraw its figures.
#
#   experiments/english-lock/reproduce.sh --dry-run    print the commands only
#   experiments/english-lock/reproduce.sh              run them (about 5 minutes per model on an M-series laptop)
#
# Needs the models in the Hugging Face cache (`steerlab doctor`). Results go to $STEERLAB_OUTPUTS/autoresearch/
# (default outputs/english-lock/).
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
export STEERLAB_OUTPUTS=${STEERLAB_OUTPUTS:-outputs/english-lock}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# The search: holds, layer x strength grid, refinement, random control and the reverse test, per model"
for model in qwen8b llama8b gemma2-9b gemma2-2b; do
  run "$PY" harness/harness.py search "$model"
done

echo "# Section 3.4: prompt S4 on Gemma-2-2B, on the search's 8 tricks and 3 questions"
run "$PY" harness/harness.py run gemma2-2b --prompt S4 --layer 8 --strength 1.5 --control

echo "# The confirmation: 24 tricks, 15 questions, 3 random directions, 95% intervals (Figure 2, Figure 3, the table)"
run "$PY" harness/confirm.py qwen8b --prompt S4 --layer 14 --strength 1.75
run "$PY" harness/confirm.py llama8b --prompt S2 --layer 15 --strength 1.25
run "$PY" harness/confirm.py gemma2-9b --prompt S3 --layer 24 --strength 4
run "$PY" harness/confirm.py gemma2-2b --prompt S4 --layer 8 --strength 1.5

echo "# Llama under prompt S4 at the same strength"
run "$PY" harness/confirm.py llama8b --prompt S4 --layer 15 --strength 1.25

echo "# The unsteered replies shown next to the steered ones (same prompt and questions, no steering)"
run "$PY" experiments/english-lock/unsteered.py qwen8b S4
run "$PY" experiments/english-lock/unsteered.py llama8b S2
run "$PY" experiments/english-lock/unsteered.py gemma2-9b S3
run "$PY" experiments/english-lock/unsteered.py gemma2-2b S4

echo "# The figures, from the shipped results/"
run "$PY" experiments/english-lock/make_figures.py
