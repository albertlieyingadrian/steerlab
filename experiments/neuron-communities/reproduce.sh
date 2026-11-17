#!/usr/bin/env bash
# Rerun the note: download the published communities, run the five GPU stages on one Modal H100 (about $3 at list
# price; needs a Modal account and Hugging Face access to Llama-3.1-8B), then the summary, figures and cover.
# With --scores, only the steps that read results/ (no GPU).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
STEPS=("$PY fetch.py")
for s in e1_base e1_instruct e2a e2b e3; do STEPS+=("$MODAL run modal_run.py --stage-name $s"); done
STEPS+=("$PY make_figures.py" "$PY make_cover.py")
[[ "${1:-}" == "--scores" ]] && STEPS=("$PY make_figures.py")
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  $step
done
