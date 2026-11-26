#!/usr/bin/env bash
# Rerun both GRAFT monitor notes (Episode 1: GRAFT with an LLM policy monitor; Episode 2: drift, generations and the
# self-monitor test). Every GPU step runs on Modal (`modal setup` first; profile via MODAL_PROFILE); the whole study
# cost about $114 of Modal GPU time on 2026-11-26. Monitor endpoints: monitors/modal_serve.py (README.md, "Monitors").
#   bash reproduce.sh --dry-run    print every command behind the headline numbers
#   bash reproduce.sh --scores     only the steps that read the shipped results/ (analysis, figures, tests)
#   bash reproduce.sh --smoke      a 3-step training smoke run plus the CPU tests (about $0.10)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$( [ -x ../../.venv/bin/python ] && echo ../../.venv/bin/python || echo python3 )}  # numpy/matplotlib/modal steps and figures
LPY=${LPY:-.venv/bin/python}               # library venv (inspect-ai, control-arena, Petri; see README.md)
# trainer tests use the trainer image's pins: uv venv -p 3.11 .venv-trainer && VIRTUAL_ENV=.venv-trainer uv pip install
#   torch==2.10.0 transformers==5.2.0 accelerate safetensors openai pytest
TPY=${TPY:-.venv-trainer/bin/python}
MODAL=${MODAL:-$( [ -x ../../.venv/bin/modal ] && echo ../../.venv/bin/modal || echo modal )}
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
T="$PY trainer/modal_train.py"
STEPS=(
  "$MODAL run modal_sample.py --model Qwen/Qwen3-0.6B --split train --n 512 --seed 2 --tag explore_base_train_v2"
  "$PY monitors/modal_serve.py deploy"
  "$LPY -m monitors.bench build"
  "$LPY -m monitors.bench score --backends rule,oss_safeguard,oss_safeguard@low,qwen_judge,control_arena,control_arena@library_default,petri"
  "$LPY -m monitors.bench summary"
  "$MODAL deploy trainer/modal_train.py"
  "PY=$PY bash launch_ep1.sh"
  "$T launch --s_per_step 12 --set name=ep2_ctrl_s11 arm=none no_hackable=true seed=11 steps=250 eval_split=heldout max_usd=5"
  "$T launch --s_per_step 12 --set name=ep2_ctrl_s12 arm=none no_hackable=true seed=12 steps=250 eval_split=heldout max_usd=5"
  "GEN=2 PY=$PY bash launch_ep2.sh   # after the ep1_* runs finish"
  "GEN=3 PY=$PY bash launch_ep2.sh   # after the ep2_g2_* runs finish"
  "$PY analyze_ep1.py --pull"
  "$LPY analyze_ep2.py --pull --score"
  "$LPY -m battery.drift --base Qwen/Qwen3-0.6B --checkpoint /vol/runs/ep1_none_s11/deployed --control /vol/runs/ep2_ctrl_s11/deployed --tag q1_none_s11 --petri 20   # likewise for each arm, seed and generation (q1_<arm>_s<seed>; q2_g<g>_<arm>_s<seed> with --no-bdiff)"
  "$MODAL run q3_selfmon.py::sample --model HuggingFaceTB/SmolLM2-1.7B-Instruct --tag smollm2-1.7b"
  "$MODAL run q3_selfmon.py::judge --judge Qwen/Qwen3-0.6B --own results/monitor_bench/trajectories.jsonl --other results/q3/episodes_smollm2-1.7b.jsonl --tag qwen3-0.6b"
  "$MODAL run q3_selfmon.py::sample --model Qwen/Qwen3-4B --tag qwen3-4b"
  "$MODAL run q3_selfmon.py::judge --judge Qwen/Qwen3-4B --own results/q3/episodes_qwen3-4b.jsonl --other results/monitor_bench/trajectories.jsonl --tag qwen3-4b"
  "$MODAL run q3_selfmon.py::judge --judge Qwen/Qwen3-4B --own results/monitor_bench/trajectories.jsonl --other results/q3/episodes_qwen3-4b.jsonl --tag qwen3-4b_swap   # post-hoc framing swap"
  "$MODAL app stop graft-monitor-serve -y; $MODAL app stop graft-monitor-train -y; $MODAL app stop graft-monitor-battery-judge -y"
)
SCORES=(
  "$PY analyze_ep1.py"
  "$LPY analyze_ep2.py"
  "$PY make_figures.py"
  "$PY make_figures_ep2.py"
  "$PY -m pytest -q -p no:cacheprovider tests/test_env.py"
  "$LPY -m pytest -q -p no:cacheprovider tests halo/tests"
  "$TPY -m pytest -q -p no:cacheprovider trainer/tests"
)
SMOKE=(
  "$MODAL deploy trainer/modal_train.py"
  "$T launch --s_per_step 12 --set name=smoke_repro arm=graft flag_source=gt steps=3 eval_every=3 eval_n=32 max_usd=1"
)
case "${1:-}" in
  --dry-run) printf '%s\n' "${STEPS[@]}" "${SCORES[@]}"; exit 0 ;;
  --scores) STEPS=("${SCORES[@]}") ;;
  --smoke) STEPS=("${SMOKE[@]}" "$LPY -m pytest -q -p no:cacheprovider tests halo/tests" "$TPY -m pytest -q -p no:cacheprovider trainer/tests") ;;
  *) STEPS+=("${SCORES[@]}") ;;
esac
for step in "${STEPS[@]}"; do echo "+ $step"; eval "$step"; done
