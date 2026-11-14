#!/usr/bin/env bash
# Rerun the note. Step 4 calls the Anthropic API (ANTHROPIC_API_KEY, from the environment or .env) through Message
# Batches; submit, wait (most batches finish within an hour), then collect. Step 5 needs a Modal account (one A10G).
# Data (about 0.8 GB of videos) goes to $PHYS_DIR, default ~/.cache/steerlab/physics-judge. With --scores, only the
# steps that read saved results: analysis and figures. Costs are in the note's "Reproduce it" section.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
STEPS=("$PY 1_sample.py" "$PY 2_traps.py" "$PY 3_heuristics.py")
for split in dev test; do
  for config in plain checklist rules motion; do
    for model in opus sonnet haiku; do
      STEPS+=("$PY 4_claude.py submit --config $config --model $model --split $split --effort medium")
    done
  done
  for rep in 1 2; do
    for model in sonnet haiku; do
      STEPS+=("$PY 4_claude.py submit --config checklist --model $model --split $split --effort medium --rep $rep")
    done
  done
done
for config in rules_noframes plain_noframes; do  # amendment 1: the text-only control
  STEPS+=("$PY 4_claude.py submit --config $config --model opus --effort medium")
done
for config in plain checklist rules motion; do
  for model in opus sonnet haiku; do
    STEPS+=("$PY 4_claude.py submit --config $config --model $model --source traps --effort medium")
  done
done
STEPS+=("$PY 4_claude.py collect" "$MODAL deploy modal_wmb.py" "$PY 5_wmb.py items" "$PY 5_wmb.py traps"
  "$PY 6_analyse.py" "$PY make_figures.py" "$PY make_gallery.py")
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
