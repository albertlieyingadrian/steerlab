#!/usr/bin/env bash
# Rerun the note: sample 120 SWE-agent trajectories (Hugging Face datasets-server, about 10 minutes, free), parse and
# label every step, run the reader (M1, plus the exploratory prior-reasoning views) and the two judges (M2, M3) with
# gpt-oss-120b through OpenRouter, score, and draw the figures and the cover. About $1 of model calls; replies are
# cached in ~/.cache/steerlab/trace-monitorability/, so a rerun costs nothing. Needs an OpenRouter key in .env
# (OPENROUTER_API_KEY). With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(1_sample.py 2_steps.py 3_reader.py "3_reader.py --prior" 4_divergence.py 5_unmentioned.py 6_score.py
       make_figures.py make_cover.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(6_score.py make_figures.py make_cover.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" $step
done
