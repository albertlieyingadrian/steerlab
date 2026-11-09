#!/usr/bin/env bash
# Rerun the note: probes on Qwen2.5-1.5B and 7B (CPU, seconds, from scalable-oversight-101's cached activations),
# four LoRA fine-tunes of Qwen2.5-7B (about an hour on an Apple-silicon laptop), the summary and the figure. Needs
# scalable-oversight-101's data and activations (its reproduce.sh) and Qwen2.5-7B-Instruct in the Hugging Face cache.
# No API calls. With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(1_probes.py 2_lora.py 3_summarise.py make_figures.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(3_summarise.py make_figures.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  "$PY" "$step"
done
