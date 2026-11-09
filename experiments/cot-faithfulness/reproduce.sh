#!/usr/bin/env bash
# Rerun the note: Qwen3-1.7B and Qwen3-8B reasoning on 300 MMLU questions in three conditions, early answering, the
# judge check and scores, the summary and the figures. About 3½ hours on an Apple-silicon laptop with both models in
# the Hugging Face cache, plus about $0.90 of judge calls (cached in ~/.cache/steerlab/cot-faithfulness/, so a rerun
# costs nothing). Needs OPENAI_API_KEY in .env for the judge. With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=("1_generate.py --model qwen3-1.7b" "2_truncate.py --model qwen3-1.7b" "1_generate.py --model qwen3-8b"
       "2_truncate.py --model qwen3-8b" 0_judge_check.py 3_score.py 4_summarise.py make_figures.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(4_summarise.py make_figures.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" $step
done
