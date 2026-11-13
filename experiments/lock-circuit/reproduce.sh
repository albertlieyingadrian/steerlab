#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/lock-circuit/reproduce.sh --dry-run    print the commands only
#   experiments/lock-circuit/reproduce.sh              run them
#
# Model work runs on rented GPUs on Modal (a B200, H200 or H100 for the graphs, an H100 for the swaps; about $13 in
# all). It needs a Modal account (MODAL_PROFILE) and, for step 3, ANTHROPIC_API_KEY (about $4 of Claude Opus 5.5).
# The lock's LoRA is the Qwen-Scope note's (experiments/tinker-qwen-scope/2_download.py puts it in
# ~/.cache/steerlab/tinker-qwen-scope/lock/adapter). The 97 GB of transcoders, the merged model and the full graphs
# stay on the Modal volume `lock-circuit`; results/ holds the small files the note reads. Set CIRCUIT_TRACER_SRC to a
# clone of circuit-tracer 0.5.3 (github.com/decoderesearch/circuit-tracer).
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/lock-circuit

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}
get() {
  run "$PY" -m modal volume get --force lock-circuit "/out/$1" "$E/results/$1"
}

echo "# 0: fold the lock's LoRA into Qwen3-8B (local check that the merge matches the adapter)"
run "$PY" "$E/0_merge.py"
echo "# 1: phase A on the GPU: behaviour, 24 graphs, the pre-registered selection, first-order attribution, controls,"
echo "#    every tested unit's activations on B-explain; then the first-token-free reconstruction"
run "$PY" -m modal run "$E/modal_app.py" --phase a
run "$PY" -m modal run "$E/modal_app.py" --phase a --steps recon
for f in behaviour.json graphs.json selection.json firstorder.json controls.json explain_meta.json explain_acts.npz recon_nopos0.json; do get "$f"; done
echo "# 2: shrink graphs.json to graphs_summary.json (the full file moves to the work folder)"
run "$PY" "$E/2_compact_graphs.py"
echo "# 3: one explanation and rule per unit, by Claude Opus 5.5, before any swap"
run "$PY" "$E/3_explain.py"
echo "# 4: phase B on the GPU: 177 swap configurations; then the exploratory patches (phases C and D)"
run "$PY" -m modal run "$E/modal_app.py" --phase b
run "$PY" -m modal run "$E/modal_app.py" --phase c
run "$PY" -m modal run "$E/modal_app.py" --phase d
for f in phase_b.json patch_exploratory.json resid_patch_exploratory.json; do get "$f"; done
echo "# 5: the pre-registered rules, and the figures"
run "$PY" "$E/4_analyze.py"
run "$PY" "$E/make_figures.py"
