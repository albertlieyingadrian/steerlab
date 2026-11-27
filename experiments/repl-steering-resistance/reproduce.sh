#!/usr/bin/env bash
# Every command behind the steering-resistance note. Costs Modal GPU time (see spend_ledger.jsonl).
# --dry-run prints the commands; with no flag only the free analysis steps run on the shipped results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export MODAL_PROFILE=credits3 MODAL_IMAGE_BUILDER_VERSION=2025.06
GPU_STEPS=(
  "../../.venv/bin/modal deploy modal_engine.py"
  "../../.venv/bin/modal deploy modal_vllm.py"
  "$PY run.py known"
  "$PY run.py known --judge2"
  "$PY run.py relevance --model 70b"
  "$PY run.py labels --model 70b --n 240"
  "$PY run.py concrete --model 70b"
  "$PY run.py calib --model 70b --grid 3,4.5,6,8,10,13,17,24"
  "$PY run.py arms --model 70b --arms B,N,M,A,R,D,T"
  "$PY run.py relevance --model 8b"
  "$PY run.py labels --model 8b --n 240"
  "$PY run.py concrete --model 8b"
  "$PY run.py calib --model 8b"
  "$PY run.py arms --model 8b --arms B,N,M"
)
for t in B70 N70 M70 A70 R70 D70 T70 B8 N8 M8; do GPU_STEPS+=("$PY run.py judge --tag $t"); done
GPU_STEPS+=("$PY run.py judge --tag B70 --judge2 --only-multi" "$PY run.py judge --tag A70 --judge2 --only-multi")
if [[ "${1:-}" == "--dry-run" ]]; then printf '%s\n' "${GPU_STEPS[@]}" "$PY analyze.py" "$PY make_figures.py"; exit 0; fi
if [[ "${1:-}" == "--gpu" ]]; then for c in "${GPU_STEPS[@]}"; do echo "+ $c"; eval "$c"; done; fi
$PY analyze.py
$PY make_figures.py
