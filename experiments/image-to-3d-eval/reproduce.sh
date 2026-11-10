#!/usr/bin/env bash
# Rerun the note. Step 1 (generation and scoring) needs a Modal account (one L40S per job; $6.48 billed on 2026-11-10,
# including image builds and failed attempts) and a Hugging Face token in a Modal secret named huggingface-secret,
# with the gated licences of stabilityai/stable-fast-3d, facebook/dinov3-vith16plus-pretrain-lvd1689m and
# black-forest-labs/FLUX.2-dev accepted. Step 4 calls OpenAI (OPENAI_API_KEY) and OpenRouter (OPENROUTER_API_KEY),
# from the environment or .env: $13.16 at list prices. Data (about 3 GB, once) goes to $I3D_DIR, default
# ~/.cache/steerlab/image-to-3d. With --scores, only the steps that read saved results: analysis and figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL_PY=${MODAL_PY:-uv run --no-project --with modal --with pandas python}
DATA_PY=${DATA_PY:-uv run --no-project --with pandas --with pyarrow --with imageio[ffmpeg] --with av --with pillow --with numpy python}
EMBED_PY=${EMBED_PY:-uv run --no-project --with torch --with open_clip_torch --with transformers --with pandas --with pillow python}
JUDGE_PY=${JUDGE_PY:-uv run --no-project --with openai --with pandas --with pillow python}
STEPS=("$MODAL_PY -m modal deploy modal_i3d.py" "$MODAL_PY 1_generate.py" "$DATA_PY 2_human_pairs.py"
  "$EMBED_PY 3_cheap_metrics.py" "$JUDGE_PY 4_judge.py human" "$JUDGE_PY 4_judge.py gso" "$PY 5_analyse.py"
  "$PY make_figures.py" "$DATA_PY make_gallery.py" "$PY make_cover.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=("$PY 5_analyse.py" "$PY make_figures.py")
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
