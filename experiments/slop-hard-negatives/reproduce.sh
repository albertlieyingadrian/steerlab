#!/usr/bin/env bash
# Rerun the hard-negative rounds note. Needs: a Modal account whose `slop-detector` volume holds the slop-detector
# note's round-0 adapter (runs/b1b-1.7bB) and data v2b (data/b1b); a local copy of data v2b ($V2B_DIR, for the
# validation labels); an OpenRouter key (open-weight models only); the EditLens clone ($EDITLENS_DIR) and Epoch AI's
# detector test ($EPOCH_DIR), as in the slop-detector note.
# --scores recomputes every number and figure from the saved results; --dry-run prints every command.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:-modal4} MODAL_IMAGE_BUILDER_VERSION=${MODAL_IMAGE_BUILDER_VERSION:-2025.06}
SCORES=(
  "$PY evaluate.py"
  "$PY interp.py"
  "$PY evaluate.py preds"
  "$PY make_figures.py"
)
STEPS=(
  "$MODAL deploy modal_hn.py"
  "$PY pools.py build"
  "$PY pools.py pull"
  "$PY gen.py known"
  "$PY pools.py interp"
  "$PY gen.py fresh"
  "$PY gen.py judge --src data/interp_corp_src.jsonl --q corporate"
  "$PY gen.py mirrors --src data/interp_slop_src.jsonl --out data/interp_slop_mirrors.jsonl"
  "$PY loop.py pilot"
  "$PY loop.py prep"
  "$PY loop.py round0"
  "$PY loop.py arm a"
  "$PY loop.py arm b"
  "$PY loop.py arm c"
  "$PY loop.py seeds"
  "$PY noise.py judge"
  "$PY noise.py sample"
  "$PY noise.py check"
)
case "${1:-}" in
  --dry-run) printf '%s\n' "${STEPS[@]}" "${SCORES[@]}"; exit 0 ;;
  --scores) RUN=("${SCORES[@]}") ;;
  *) RUN=("${STEPS[@]}" "${SCORES[@]}") ;;
esac
for step in "${RUN[@]}"; do
  echo "+ $step"
  eval "$step"
done
