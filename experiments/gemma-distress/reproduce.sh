#!/usr/bin/env bash
# Rerun the note: the judge check, 13 models x 80 repeated-rejection conversations, the judge, two fine-tunes of
# Gemma-3-1B-it (plain DPO, and DPO with an imitation term) with their conversations and GSM8K check, the summary and
# the figures. About 2 hours on an Apple-silicon laptop with every model in the Hugging Face cache; needs
# OPENAI_API_KEY in .env for the judge (about $1.30; cached in ~/.cache/steerlab/gemma-distress/, so a rerun costs
# nothing). With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(0_judge_check.py "1_converse.py --all" 2_judge.py 3_dpo.py 4_after.py 2_judge.py 5_summarise.py make_figures.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(5_summarise.py make_figures.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" $step
done
