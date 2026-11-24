#!/usr/bin/env bash
# Rerun "Which slop rules actually separate AI from human text?". CPU only; no GPU.
# Needs: the EditLens repo's data (pangram/editlens_iclr; research licence, gated) in ~/.cache/steerlab/editlens/data
# (test.csv, val.csv, test_enron.csv, raid_10k.csv), HC3's all.jsonl (huggingface.co/datasets/Hello-SimpleAI/HC3,
# CC BY-SA 4.0) in ~/.cache/steerlab/slop-rules/hc3/, and an OpenRouter key in OPENROUTER_API_KEY_2 for the generation
# step (open-weights models only; hard cap $15 in generate.py; this run cost about $6).
# With --scores, only the steps after generation (they read the cached generations).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(
  "$PY prompts.py"
  "$PY generate.py"
  "$PY corpus.py"
  "$PY evaluate.py"
  "$PY discover.py"
  "$PY refit.py lists"
  "$PY refit.py fit"
  "$PY compare.py"
  "$PY notes.py"
  "$PY predictions.py"
  "$PY make_figures.py"
)
if [[ "${1:-}" == "--scores" ]]; then
  STEPS=("${STEPS[@]:2}")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "== $step"
  eval "$step"
done
