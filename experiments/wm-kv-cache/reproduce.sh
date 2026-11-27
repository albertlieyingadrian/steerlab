#!/usr/bin/env bash
# Rerun the note. Training and evaluation run on Modal (one A10G per seed, at most two at once, about 25 minutes per
# seed, under $2 in all); the untrained floor, the summary, the figures and the cover run locally on CPU.
# With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
STEPS=(
  "$MODAL run modal_app.py::train_seeds --seeds 1,2 --steps 6000"
  "$MODAL run modal_app.py::train_seeds --seeds 3 --steps 6000"
  "$MODAL run modal_app.py::eval_seeds --seeds 1,2"
  "$MODAL run modal_app.py::eval_seeds --seeds 3"
  "$PY train.py --seed 1 --steps 0 --out results/ckpt_untrained_s1.pt --device cpu"
  "$PY evaluate.py --ckpt results/ckpt_untrained_s1.pt --out results/eval_untrained_s1.json --device cpu --only sweep"
  "$PY analyse.py"
  "$PY make_figures.py"
  "$PY make_cover.py"
)
[[ "${1:-}" == "--scores" ]] && STEPS=("$PY analyse.py" "$PY make_figures.py")
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  eval "$step"
done
