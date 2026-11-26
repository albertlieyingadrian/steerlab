#!/usr/bin/env bash
# Rerun the slop-detector self-improvement pilots. Needs: a Modal account whose `slop-detector` volume holds the
# slop-detector note's round-0 adapter (runs/b1b-1.7bB) and data v2b; the hard-negative note's data folder (pools,
# round-0 pool scores, mirrors, frozen evaluation file); the HC3 and RAID sets; macOS (sandbox-exec).
# --figures redraws every figure and prints every headline number from results/; --dry-run prints every command.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:-modal4} MODAL_IMAGE_BUILDER_VERSION=${MODAL_IMAGE_BUILDER_VERSION:-2025.06}
FIGURES=(
  "$PY make_figures.py"
)
STEPS=(
  "$MODAL deploy modal_rsi.py"
  "$MODAL deploy modal_llm.py"
  "$PY prepare.py pools"
  "$PY prepare.py decontam"
  "$PY prepare.py upload"
  "$PY prepare.py candidates"
  "$PY score.py cand"
  "$PY prepare.py freeze 1000"
  "$PY prepare.py upload_p2"
  "$PY score.py ref"
  "$PY loop.py smoke"
  "$PY loop.py noise"
  "$PY loop.py l2"
  "$PY loop.py l3"
  "$PY analyze.py"
)
case "${1:-}" in
  --dry-run) printf '%s\n' "${STEPS[@]}" "${FIGURES[@]}"; exit 0 ;;
  --figures) RUN=("${FIGURES[@]}") ;;
  "") RUN=("${STEPS[@]}" "${FIGURES[@]}") ;;
  *) echo "usage: $0 [--dry-run|--figures]" >&2; exit 2 ;;
esac
for cmd in "${RUN[@]}"; do
  echo "+ $cmd"
  eval "$cmd"
done
