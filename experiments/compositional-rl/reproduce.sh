#!/usr/bin/env bash
# Every command behind the compositional-rl note, in order. --dry-run prints them without running.
# Steps 1-5 cost money (Modal H100s, profile set in MODAL_PROFILE); steps 6-8 run on a laptop from the shipped results.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
MODAL=${MODAL:-.venv/bin/modal}
E=experiments/compositional-rl
run() { echo "+ $*"; if [[ "${1-}" != "--dry" && "${DRY:-0}" != 1 ]]; then "$@"; fi; }
[[ "${1-}" == "--dry-run" ]] && DRY=1

# 1. untrained-model screen (chose the model and lengths before the pre-registration)
run $MODAL run $E/modal_rl.py::screen --model Qwen/Qwen3-1.7B
run $MODAL run $E/modal_rl.py::screen --model Qwen/Qwen2.5-1.5B-Instruct
# 2. learning-rate pilot (training reward only)
run $MODAL run $E/modal_rl.py::train_many --config configs/pilot_lr.json
# 3. untrained-model evaluation, then the 22 pre-registered runs (one H100 each, at most 4 at once)
for L in 1 2 3; do run $MODAL run $E/modal_rl.py::eval_base --config configs/l$L.json --tag base-l$L; done
run $MODAL run $E/modal_rl.py::train_many --config configs/all.json
# 4. amendment 1: the Level III RL-Compound seed 0 rerun with micro-batch 2
run $MODAL run $E/modal_rl.py::train_many --config configs/l3_rerun_s0.json
# 5. the step test and the adapter swaps
run $MODAL run $E/modal_rl.py::mechanism --jobs configs/mech_step.json --split 2
run $MODAL run $E/modal_rl.py::mechanism --jobs configs/mech_swap.json --split 2
# 6-8. analysis, figures, cover
run $PY $E/analyse.py
run $PY $E/make_figures.py
run $PY $E/make_cover.py
