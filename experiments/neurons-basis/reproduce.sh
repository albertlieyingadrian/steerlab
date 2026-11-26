#!/usr/bin/env bash
# Every command behind the numbers in report/index.html. `./reproduce.sh --dry-run` prints them.
# Model stages run on Modal (MODAL_PROFILE set; they cost money: see the plan's budget); the rest runs on a laptop.
# Needs: neuronkit (pip install "neuronkit[labels] @ git+https://github.com/albertlieyingadrian/neuronkit"),
# the detector-sanity-checks work folder for the organism rows and residual activations (its extract_organism.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
# neuronkit: pip install it, or point NEURONKIT_PATHS / NEURONKIT_SRC at a source checkout
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" != 1 ]]; then "$@"; fi; }

E=experiments/neurons-basis
run modal deploy $E/modal_app.py
# Phase 1: percentiles, sparsity, benchmarks, neurons (subcircuits, clusters, damage), per model
run $PY $E/launch.py phase1 qwen05 sparsity,bench,neurons
run $PY $E/launch.py phase1 olmoe sparsity,bench,neurons
# download neurons_judge.json per model into ~/.cache/steerlab/neurons-basis/<model>/, then the judge
run $PY $E/judge_phase1.py check qwen05 --thinking --true-labels
run $PY $E/judge_phase1.py check qwen05 --thinking
run $PY $E/judge_phase1.py run qwen05 --thinking
run $PY $E/judge_phase1.py run olmoe --thinking
run $PY $E/judge_phase1.py crosscheck qwen05 --n 30
# upload judge_out.json to the volume, then simulation patching and the writing test
run $PY $E/launch.py phase1 qwen05 patch
run $PY $E/launch.py phase1 olmoe patch
run $PY $E/analyze_phase1.py
# Amendment 8: the original study's rule and benchmark suite (lm-evaluation-harness)
run $PY $E/launch.py phase35_big qwen36
run $PY $E/launch.py phase35_small olmoe
# Phase 2: the password-locked organism
run $PY $E/launch.py phase2 extract,pct
run $PY $E/monitors_phase2.py
run $PY $E/launch.py phase2 validate,explain,examples
run $PY $E/judge_phase1.py run organism_examples --thinking
run $PY $E/zeroshot_phase2.py
run $PY $E/summarize_phase2.py
run $PY $E/validate_circuitcheck.py
run $PY $E/analyze_phase35.py
# figures and cover
run $PY $E/make_figures.py
run $PY $E/make_cover.py
