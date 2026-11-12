#!/usr/bin/env bash
# Rerun the slop-detector note. Needs a Hugging Face token (generation through Inference Providers, read by
# huggingface_hub), a Modal account (training and scoring), a clone of github.com/pangramlabs/EditLens with its data at
# $EDITLENS_DIR (default ~/.cache/steerlab/editlens; EditLens is CC BY-NC-SA 4.0, non-commercial) and a clone of
# github.com/jaeholee-brown/ai-text-detectors at $EPOCH_DIR (default ~/.cache/steerlab/epoch-detectors/repo).
# --scores recomputes every number and figure from the saved scores only; --dry-run prints every command.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_IMAGE_BUILDER_VERSION=${MODAL_IMAGE_BUILDER_VERSION:-2025.06}
SCORES=(
  "$PY 05_eval.py --run runs/pilot-0.6b"
  "$PY 05_eval.py --run runs/pilot-1.7b"
  "$PY 05_eval.py --run runs/v2-1.7b"
  "$PY 05_eval.py --run runs/v2-4b"
  "$PY tune_bands.py test"
  "$PY tune_editlens_val.py test"
  "$PY tune_editlens_val.py control"
  "$PY make_figures.py"
)
STEPS=(
  "$MODAL deploy modal_slop.py"
  "$PY score_voi.py"
  "$PY 01_human.py"
  "$PY 02_generate.py --budget 11"
  "$PY 03_label.py"
  "$PY train_modal.py --data pilot --runs 0.6b=Qwen/Qwen3-0.6B-Base 1.7b=Qwen/Qwen3-1.7B-Base"
  "$PY 01_human.py --only amazon arxiv hn medium reddit stack xsum --offset 1600 --per-source 1500 --suffix _v2"
  "$PY 01_human.py --only medium xsum --offset 3100 --per-source 1000 --max-words 550 --suffix _long"
  "$PY 01_human.py --only pg19 --per-source 1500 --max-words 550"
  "$PY 02_generate.py --plan v2 --imitations 3000 --budget 18.85"
  "$PY 03_label.py"
  "$PY train_modal.py --data pilotR --runs 1.7b=Qwen/Qwen3-1.7B-Base --splits"
  "$PY train_modal.py --data v2 --runs 1.7b=Qwen/Qwen3-1.7B-Base --splits"
  "$PY compare_val.py --old pilotR-1.7b --new v2-1.7b"
  "$PY score_run.py v2-1.7b"
  "$PY train_modal.py --data v2 --gpu a100 --runs 4b=Qwen/Qwen3.5-4B --extra --epochs 2 --grad-ckpt --ckpt-minutes 35 --max-minutes 300 --resume"
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
