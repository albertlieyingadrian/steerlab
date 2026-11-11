#!/usr/bin/env bash
# Rerun the note. Needs uv, node 18+, a Modal account (an L40S for about 3 GPU-hours in all, about $9.50 billed on
# 2026-11-10; CPU for the GGUF export) and an OpenAI key for the judge, voice rewrites and cover (about $2.70), read from the
# environment or .env. slop-score (MIT) is cloned at a pinned commit to $SLOP_SCORE_DIR, default
# ~/.cache/steerlab/slop-score. The EditLens steps need access to pangram/editlens_roberta-large and
# pangram/editlens_iclr on Hugging Face (CC BY-NC-SA 4.0, non-commercial, by request), a clone of
# github.com/pangramlabs/EditLens at $EDITLENS_DIR (default ~/.cache/steerlab/editlens) and the `emoji` package at
# $EDITLENS_PYLIBS; editlens_qwen.py trains on one H100 (about $3); voi_slop.py scores on one L4 or A10G (about $0.12). With --scores, only the steps that read saved replies: band, analysis, figures.
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
  "$PY 2_voices.py"
  "$PY voice_eval.py check"
  "$PY voice_eval.py rewrite"
  "$PY voice_eval.py score"
  "$PY detect.py mirrors"
  "$PY detect.py score"
  "$PY editlens.py check"
  "$PY editlens.py score"
  "$PY editlens_qwen.py"
  "$MODAL deploy modal_voi_slop.py"
  "$PY voi_slop.py score"
  "$PY analyze.py"
  "$PY make_figures.py"
  "$PY make_cover.py"
  "$PY release.py --export"
  "$PY release.py"
)
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=("$PY 1_band.py" "$PY voice_eval.py score" "$PY editlens.py score" "$PY analyze.py" "$PY make_figures.py")
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
