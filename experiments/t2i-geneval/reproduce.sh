#!/usr/bin/env bash
# Rerun the note. Needs a Modal account (run `modal deploy modal_app.py` first, with a Hugging Face token stored as the
# Modal secret `huggingface-secret`) and OPENROUTER_API_KEY in the environment or .env for the planner. A full rerun
# costs about $45 of Modal and under $1 of planner calls at 2026-11 list prices. Every step caches by image path and
# resumes. The blind look needs a person: `blind_look.py make` writes the sheets, the rater writes
# results/blind/ratings.json, then `blind_look.py score` unblinds. With --scores, only the steps that read saved results.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(
  "$PY run_eval.py --set official --samples 4 --size 1024 --indep --run schnell_official"
  "$PY harness.py --set val --reps 2 --retry"
  "$PY harness.py --set official --reps 4 --variants plan --nmax 4 --retry"
  "$PY run_eval.py --set compbench --samples 1 --size 1024 --run cb_base1024"
  "$PY run_eval.py --set compbench --samples 1 --size 1024 --plan --run cb_plan1024"
  "$PY guard_genai.py"
  "$PY latency.py --n 20"
  "$PY blind_look.py make"
)
SCORES=("$PY blind_look.py score" "$PY pilot_summary.py" "$PY make_figures.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=()
fi
STEPS+=("${SCORES[@]}")
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
