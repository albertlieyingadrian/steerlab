#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/context-directions/reproduce.sh --dry-run    print the commands only
#   experiments/context-directions/reproduce.sh              run them (about 1.5 hours on an M-series laptop)
#
# Downloads Qwen3-1.7B and Qwen3-4B from Hugging Face, and the two public prompt sources below. Activations (about
# 1 GB per model) go to the lab's outputs folder or ~/.cache; results/ holds every number and reply.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-../../.venv/bin/python}
SRC=${SRC:-$(mktemp -d)}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 0: the prompt sets (AdvBench and Alpaca, downloaded into a fresh folder)"
run curl -sSL -o "$SRC/harmful_behaviors.csv" https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
run curl -sSL -o "$SRC/alpaca_data.json" https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
run "$PY" build_data.py --advbench "$SRC/harmful_behaviors.csv" --alpaca "$SRC/alpaca_data.json"
echo "# 1: activations, geometry, strength sweep and transfer grid (pre-registered)"
run "$PY" study.py extract
run "$PY" study.py geometry
run "$PY" study.py steer-dev
run "$PY" study.py steer
echo "# 2 (exploratory): control contexts, whitening, truth on unused capitals, the reply-based Chinese direction"
run "$PY" study.py extract-controls
run "$PY" study.py geometry-controls
run "$PY" robustness.py
run "$PY" study.py steer-truth-capitals
run "$PY" chinese_reply.py gen
run "$PY" chinese_reply.py run
echo "# 3 (exploratory): the geometry on Qwen3-4B (layer 18 of 36)"
run env CTX_MODEL=Qwen/Qwen3-4B CTX_TAG=_qwen3_4b CTX_LAYER=18 "$PY" study.py extract
run env CTX_MODEL=Qwen/Qwen3-4B CTX_TAG=_qwen3_4b CTX_LAYER=18 "$PY" study.py extract-controls
run env CTX_MODEL=Qwen/Qwen3-4B CTX_TAG=_qwen3_4b CTX_LAYER=18 "$PY" study.py geometry
run env CTX_MODEL=Qwen/Qwen3-4B CTX_TAG=_qwen3_4b CTX_LAYER=18 "$PY" study.py geometry-controls
echo "# 4: verdicts, every quoted number, figures and cover"
run "$PY" summary.py
run env CTX_TAG=_qwen3_4b CTX_LAYER=18 "$PY" summary.py
run "$PY" make_figures.py
run env CTX_TAG=_qwen3_4b CTX_LAYER=18 "$PY" make_figures.py
run "$PY" make_cover.py
echo "# The refusal hand check (results/refusal_handcheck.json) was done by reading 40 replies; it is not rerun."
