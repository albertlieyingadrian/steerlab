#!/usr/bin/env bash
# Rerun the note: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct answer 407 one-word prompts, predict properties of
# their own and each other's answers (no training, then after LoRA fine-tuning), the summary and the figure. About
# 45 minutes on an Apple-silicon laptop with both models in the Hugging Face cache. No API calls.
# With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(1_object.py 2_predict.py 3_finetune.py 4_summarise.py make_figures.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(4_summarise.py make_figures.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  "$PY" "$step"
done
