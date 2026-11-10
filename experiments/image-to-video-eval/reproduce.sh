#!/usr/bin/env bash
# Rerun the note. Steps 3 and 4 need a Modal account (H100 80 GB for generation, L4 or A10G for scoring) and a Hugging
# Face token in a Modal secret named huggingface-secret. Step 5 calls OpenAI (OPENAI_API_KEY) and OpenRouter
# (OPENROUTER_API_KEY), from the environment or .env. Costs are in the note's "Reproduce it" section. Data (about 3 GB,
# once) goes to $I2V_DIR, default ~/.cache/steerlab/image-to-video. With --scores, only the steps that read saved
# results: analysis and figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL_PY=${MODAL_PY:-uv run --no-project --with modal --with pandas --with pillow --with numpy python}
DATA_PY=${DATA_PY:-uv run --no-project --with pandas --with pyarrow --with pillow --with numpy python}
JUDGE_PY=${JUDGE_PY:-uv run --no-project --with openai --with pandas --with pillow python}
STEPS=("$DATA_PY 1_human_pairs.py" "$DATA_PY 2_traps.py" "$MODAL_PY -m modal deploy modal_i2v.py"
  "$MODAL_PY 3_generate.py" "$MODAL_PY 4_scores.py human" "$MODAL_PY 4_scores.py traps" "$MODAL_PY 4_scores.py ours"
  "$JUDGE_PY 5_judge.py human" "$JUDGE_PY 5_judge.py traps" "$JUDGE_PY 5_judge.py ours" "$PY 6_analyse.py"
  "$PY make_figures.py" "$DATA_PY make_gallery.py" "$PY make_cover.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=("$PY 6_analyse.py" "$PY make_figures.py")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
