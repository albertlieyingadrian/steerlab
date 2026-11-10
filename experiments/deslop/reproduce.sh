#!/usr/bin/env bash
# Rerun the note. Needs uv, node 18+, a Modal account (an L40S for about 2.5 GPU-hours in all, about $8 billed on
# 2026-11-10; CPU for the GGUF export) and an OpenAI key for the judge and the cover (about $2), both read from the
# environment or .env. slop-score (MIT) is cloned at a pinned commit to $SLOP_SCORE_DIR, default
# ~/.cache/steerlab/slop-score. With --scores, only the steps that read saved replies: band, analysis, figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
SLOP_SCORE_DIR=${SLOP_SCORE_DIR:-$HOME/.cache/steerlab/slop-score}
SLOP_SCORE_COMMIT=289264ab2a0df1358ef81ced3c259610ea709514
export SLOP_SCORE_DIR MODAL_IMAGE_BUILDER_VERSION=2025.06
STEPS=(
  "$PY 0_human.py"
  "$PY 1_band.py"
  "$MODAL deploy modal_deslop.py"
  "$PY generate.py base-train plain-train base plain sampler neurons neurons-random kannike plain+sampler plain+neurons"
  "$PY tune_steer.py"
  "$PY generate.py steer steer-random plain+steer"
  "$PY teacher.py generate"
  "$PY teacher.py filter"
  "$PY teacher.py train"
  "$PY generate.py lora"
  "$PY teacher.py round2"
  "$PY generate.py lora-r2"
  "$PY damage.py base plain sampler neurons neurons-random steer steer-random plain+neurons plain+steer plain+sampler kannike lora lora-r2"
  "$PY judge.py check"
  "$PY judge.py check --question human"
  "$PY judge.py check --question human --model gpt-5.4"
  "$PY judge.py runs plain sampler neurons neurons-random steer steer-random plain+neurons plain+steer plain+sampler kannike lora lora-r2"
  "$PY analyze.py"
  "$PY make_figures.py"
  "$PY make_cover.py"
  "$PY release.py --export"
  "$PY release.py"
)
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=("$PY 1_band.py" "$PY analyze.py" "$PY make_figures.py")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "git clone https://github.com/sam-paech/slop-score $SLOP_SCORE_DIR && git -C $SLOP_SCORE_DIR checkout $SLOP_SCORE_COMMIT && npm --prefix $SLOP_SCORE_DIR install --ignore-scripts"
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
if [[ ! -d "$SLOP_SCORE_DIR" ]]; then
  git clone -q https://github.com/sam-paech/slop-score "$SLOP_SCORE_DIR"
  git -C "$SLOP_SCORE_DIR" checkout -q "$SLOP_SCORE_COMMIT"
  npm --prefix "$SLOP_SCORE_DIR" install --ignore-scripts --no-audit --no-fund
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
