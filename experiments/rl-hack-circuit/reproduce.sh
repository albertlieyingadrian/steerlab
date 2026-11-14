#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/rl-hack-circuit/reproduce.sh --dry-run    print the commands only
#   experiments/rl-hack-circuit/reproduce.sh              run them
#
# Model work runs on rented GPUs on Modal (H100s; a few dollars). It reads Pigment Gym's Qwen3-1.7B checkpoints from
# the Modal volume `pigment-gym-qwen3` (made by experiments/pigment-gym/modal_qwen3.py) and needs ANTHROPIC_API_KEY
# for step 3. Set CIRCUIT_TRACER_SRC to a clone of circuit-tracer 0.5.3 (github.com/decoderesearch/circuit-tracer).
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/rl-hack-circuit

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}
get() {
  run "$PY" -m modal volume get --force rl-hack-circuit "out/$1" "$E/results/$1"
}

echo "# 1: download, merge every checkpoint, the step-0 check, base sketches and paste prefixes"
run "$PY" -m modal run "$E/modal_app.py" --phase prep
echo "# 2: 64 graphs (one container per contrast), the selection, first-order attribution, controls, B-explain activations"
run "$PY" -m modal run "$E/modal_app.py" --phase graphs-par
for f in prep.json graphs_summary.json selection.json firstorder.json controls.json explain_acts.npz explain_meta.json; do get "$f"; done
echo "# 3: one explanation and rule per unit, by Claude Opus 5.5, before any swap"
run "$PY" "$E/3_explain.py"
echo "# 4: swaps and causal checks; hack rates under ablation; exploratory residual patching"
run "$PY" -m modal run "$E/modal_app.py" --phase swaps
run "$PY" -m modal run "$E/modal_app.py" --phase gen
run "$PY" -m modal run "$E/modal_app.py" --phase patch
for f in swaps.json gen.json patch.json; do get "$f"; done
echo "# 5: the pre-registered rules, and the figures"
run "$PY" "$E/4_analyze.py"
run "$PY" "$E/make_figures.py"
run "$PY" "$E/make_cover.py"   # the cover, hero.jpg and the research card (local Chromium)
