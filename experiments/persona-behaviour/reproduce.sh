#!/usr/bin/env bash
# Rerun the note: Qwen2.5-7B-Instruct under six system prompts on 400 MMLU questions (pushback, a professor's hint,
# three stated questions) and 20 impossible tasks, the judge and its check, the summary and the figures. Needs
# Qwen2.5-7B-Instruct in the Hugging Face cache and OPENAI_API_KEY in .env for the judge (about $0.05).
# With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(1_measure.py 0_judge_check.py 2_judge.py 3_summarise.py make_figures.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(3_summarise.py make_figures.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" $step
done
