#!/usr/bin/env bash
# Rerun the backbone bake-off. Needs the slop-detector note's inputs (its data v2 with 4 buckets at $SLOP_DATA, the
# EditLens clone at $EDITLENS_DIR, Epoch AI's detector test at $EPOCH_DIR; see ../slop-detector/reproduce.sh), a Modal
# account with a Hugging Face secret named huggingface-secret (Llama-3.1-8B is gated) and the slop-detector volume.
# --scores recomputes every number and figure from the saved scores only; --dry-run prints every command.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_IMAGE_BUILDER_VERSION=${MODAL_IMAGE_BUILDER_VERSION:-2025.06}
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
SCORES=(
  "$PY analyze.py"
  "$PY make_figures.py"
)
STEPS=(
  "$MODAL deploy modal_bb.py"
  "$PY run.py upload"
  "$PY run.py sweep"
  "$PY run.py choose"
  "$PY run.py train"
  "$PY run.py score"
  "$PY run.py refs --ref qwen3.5-9b"
  "MODAL_PROFILE=modal4 $PY run.py refs --ref qwen3.5-4b"
  "${SCORES[@]}"
)
case "${1:-}" in
  --dry-run) printf '%s\n' "${STEPS[@]}" ;;
  --scores) for c in "${SCORES[@]}"; do echo "+ $c"; eval "$c"; done ;;
  *) for c in "${STEPS[@]}"; do echo "+ $c"; eval "$c"; done ;;
esac
