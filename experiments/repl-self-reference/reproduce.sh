#!/usr/bin/env bash
# Every command behind the self-referential prompting note. Costs Modal GPU time (see spend_ledger.jsonl).
# --dry-run prints the commands; with no flag only the free analysis steps run on the shipped results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export MODAL_PROFILE=credits3 MODAL_IMAGE_BUILDER_VERSION=2025.06
GPU_STEPS=(
  "../../.venv/bin/modal deploy ../repl-steering-resistance/modal_engine.py"
  "../../.venv/bin/modal deploy ../repl-steering-resistance/modal_vllm.py"
  "$PY run.py known"
  "$PY run.py features"
  "$PY run.py exp1"
  "cat data/jobs/exp1_*.jsonl > data/jobs/exp1.jsonl; cat results/gen_exp1_*.jsonl > results/gen_exp1.jsonl"
  "$PY run.py judge --tag exp1"
  "$PY run.py pilot --ks 2,4,8,16"
  "$PY run.py exp2"
  "$PY run.py judge --tag exp2"
  "$PY run.py tqa"
  "$PY run.py judge --tag tqa --kind tqa"
  "$PY run.py exp3"
  "$PY run.py embed"
  "$PY run.py exp4"
  "cat data/jobs/exp4_*.jsonl > data/jobs/exp4.jsonl; cat results/gen_exp4_*.jsonl > results/gen_exp4.jsonl"
  "$PY run.py judge --tag exp4 --kind aware"
)
if [[ "${1:-}" == "--dry-run" ]]; then printf '%s\n' "${GPU_STEPS[@]}" "$PY analyze.py" "$PY make_figures.py"; exit 0; fi
if [[ "${1:-}" == "--gpu" ]]; then for c in "${GPU_STEPS[@]}"; do echo "+ $c"; eval "$c"; done; fi
$PY analyze.py
$PY make_figures.py
