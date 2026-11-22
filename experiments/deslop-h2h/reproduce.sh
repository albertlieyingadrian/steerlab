#!/usr/bin/env bash
# Rerun De-slopping, Episode 2. Needs uv, node 18+, a Modal account (H100s; see results/spend.jsonl for the cost),
# Hugging Face access to google/gemma-3-27b-it (the judge) as the Modal secret `huggingface-secret`, slop-score cloned
# at Episode 1's pinned commit in $SLOP_SCORE_DIR (default ~/.cache/steerlab/slop-score), the slop-detector app with
# run f9-9b on its volume (experiments/slop-detector), Episode 1's app `deslop` deployed (its Detectors class), and
# probes-vs-interp's layer-18 units (`causal.py units` -> its data/causal_units.npz; set $PROBE_UNITS).
# Track B needs nothing new: it re-scores Episode 1's saved replies in ../deslop/results/gen/.
# With --scores, only the steps that read saved replies: detectors (cached), the table and the figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian} MODAL_IMAGE_BUILDER_VERSION=2025.06
RANDOM_ARMS="random-register-0 random-register-1 random-register-2 random-probe-0 random-probe-1 random-probe-2 random-sae-dir-0 random-sae-dir-1 random-sae-dir-2"
STEPS=(
  "$MODAL deploy modal_ep2.py"
  "$MODAL deploy ../deslop/modal_deslop.py"
  "$PY gen.py original-train plain-train teacher-train original plain sampler plain+sampler --parallel 7"
  "$PY units.py"
  "$PY tune.py"
  "$PY train.py filter"
  "$PY train.py sft"
  "$PY train.py dpo"
  "$PY gen.py register probe sae-dir sae-abl plain+register plain+sae sft dpo sft+sampler $RANDOM_ARMS --parallel 8"
  "$PY drift.py original plain register probe sae-dir sae-abl plain+register plain+sae sft dpo $RANDOM_ARMS"
  "$PY rulers.py"
  "$PY pangram.py"
  "$PY pairwise.py check"
  "$PY pairwise.py runs"
  "$PY table.py"
  "$PY make_figures.py"
  "$PY make_cover.py"
)
if [[ "${1:-}" == "--scores" ]]; then
  STEPS=("$PY rulers.py" "$PY table.py" "$PY make_figures.py")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
