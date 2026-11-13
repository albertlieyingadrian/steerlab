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
  "$PY fresh_test.py baseline"
  "$PY final_look.py look"
  "$PY make_figures.py"
  "$PY make_cover.py"
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
  "$PY fresh_test.py freeze"
  "$PY fresh_test.py score"
  "$PY relabel_b1.py"
  "$PY train_modal.py --data b1a --src data/v2b --runs 1.7bA=Qwen/Qwen3-1.7B-Base --splits editlens_val"
  "$PY train_modal.py --data b1b --src data/v2b --runs 1.7bB=Qwen/Qwen3-1.7B-Base --splits editlens_val --extra --head buckets"
  "$PY eval_b1.py --base v2-1.7b --arms b1a-1.7bA b1b-1.7bB"
  "$PY train_modal.py --data b1x --src data/v2b --gpu a100 --runs 4bk=Qwen/Qwen3.5-4B --splits editlens_val --extra --epochs 2 --grad-ckpt --head buckets --ckpt-minutes 35 --max-minutes 300 --resume"
  "$PY eval_b1.py --base v2-4b --arms b1x-4bk"
  "$PY gen_v3.py humans"
  "$PY gen_v3.py plan"
  "$PY gen_v3.py submit --editor haiku"
  "$PY gen_v3.py submit --editor sonnet"
  "$PY gen_v3.py collect"
  "$PY gen_v3.py splice"
  "$PY gen_v3.py submit --editor haiku"
  "$PY gen_v3.py collect"
  "$PY gen_v3.py label"
  "$PY train_modal.py --data b3a --src data/v3b --runs 1.7bB=Qwen/Qwen3-1.7B-Base --splits editlens_val --extra --epochs 2.941 --head buckets"
  "$PY eval_b1.py --base b1b-1.7bB --arms b3a-1.7bB --out b3_stage1"
  "$PY train_modal.py --data f9 --src data/v2b --gpu a100 --runs 9b=Qwen/Qwen3.5-9B --splits editlens_val --extra --epochs 2 --grad-ckpt --head buckets --ckpt-minutes 35 --max-minutes 600 --resume"
  "$PY train_modal.py --data f4s2 --src data/v2b --gpu a100 --runs 4bs2=Qwen/Qwen3.5-4B --splits editlens_val --extra --epochs 2 --grad-ckpt --head buckets --ckpt-minutes 35 --max-minutes 600 --resume --seed 1"
  "$PY final_look.py select"
  "$PY final_look.py score --runs f9-9b"
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
