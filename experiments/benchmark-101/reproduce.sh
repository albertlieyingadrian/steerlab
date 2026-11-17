#!/usr/bin/env bash
# Rerun the note. Needs: Modal (four H100 runs), OPENAI_API_KEY and FIREWORKS_API_KEY in the environment, and a local
# copy of the human passages (HUMAN=path/to/human.jsonl, built by experiments/slop-detector/01_human.py). About $7.
# With --scores, only the steps that read results/ (no model calls).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
HUMAN=${HUMAN:-../slop-detector/data/human.jsonl}
STEPS=("$PY -c 'import bench; bench.mmlu_redux()'")
for k in qwen1.5b qwen7b llama8b gemma9b; do STEPS+=("$MODAL run modal_a.py --key $k"); done
STEPS+=("$PY a_api.py" "$PY b_build.py --human $HUMAN" "$PY b_judge.py" "$PY b_baselines.py" "$PY analyze.py" "$PY make_figures.py" "$PY make_cover.py")
[[ "${1:-}" == "--scores" ]] && STEPS=("$PY analyze.py" "$PY make_figures.py")
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  eval "$step"
done
