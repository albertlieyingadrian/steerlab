#!/usr/bin/env bash
# Rerun the detector-evasion study. Needs a Modal account (H100, L40S, A100; see results/spend.jsonl), Hugging Face
# access to google/gemma-3-27b-it as the Modal secret `huggingface-secret`, the slop-detector volume with run f9-9b and
# its data/f9 (experiments/slop-detector), Episode 1's app `deslop` deployed (its Detectors class; needs
# MODAL_IMAGE_BUILDER_VERSION=2025.06), and Episode 3's results (experiments/deslop-h2h).
# With --scores, only the steps that read saved scores: the table, the hack catalogue and the figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
STEPS=(
  "$MODAL deploy modal_app.py"
  "env MODAL_IMAGE_BUILDER_VERSION=2025.06 $MODAL deploy ../deslop/modal_deslop.py"
  "$PY prep.py sources"
  "$PY prep.py claims"
  "$PY check_rulers.py"
  "$PY train.py pilot"
  "$PY train.py launch"
  "$PY train.py pull"
  "$PY evaluate.py gen"
  "$PY evaluate.py score"
  "$PY evaluate.py transfer"
  "env MODAL_PROFILE=modal4 $PY evaluate.py encoder"
  "$PY evaluate.py judge"
  "$PY defend.py gen"
  "$PY defend.py train"
  "$PY defend.py score"
  "$PY table.py"
  "$PY readcheck.py check"
  "$PY readcheck.py sample"
  "$PY readcheck.py read"
  "$PY readcheck.py score"
  "$PY hacks.py"
  "$PY make_figures.py"
  "$PY make_cover.py"
)
if [[ "${1:-}" == "--scores" ]]; then
  STEPS=("$PY table.py" "$PY hacks.py" "$PY make_figures.py")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for s in "${STEPS[@]}"; do
  echo "== $s"
  $s
done
