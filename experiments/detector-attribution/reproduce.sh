#!/usr/bin/env bash
# Every command behind the note, in order.
#   ./reproduce.sh            run everything (Modal + Pangram)
#   ./reproduce.sh --scores   recompute the summary and figures from results/ only (no paid calls)
#   ./reproduce.sh --dry-run  print the commands
# Needs: the deslop human passages (DESLOP_HUMAN or a steerlab checkout with experiments/deslop/data/human/),
# the slop-detector Modal volume (run f9-9b), the 1.7B adapter on the detector-attribution volume, PANGRAM_API_KEY.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
MODAL=${MODAL:-modal}
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}

steps=(
  "$MODAL deploy modal_attr.py"
  "$PY attribute.py score"          # each detector's score on the 847 cut texts
  "$PY attribute.py words"          # leave-one-word-out occlusion, all four detectors (primary attribution)
  "$PY attribute.py occlude"        # the 9B's sentence occlusion on confirmation AI texts
  "$PY attribute.py deletion"       # top vs 20 random deletions, scored by all four
  "$PY attribute.py ig"             # integrated gradients on 160 texts (secondary)
  "$PY sae.py attr"                 # Qwen-Scope features inside the 9B, discovery texts
  "$PY sae.py ablate"               # top-10 / top-50 vs 20 matched random sets, confirmation texts
  "$PY sae.py label"                # where the top features fire
  "$PY gen.py means"                # the writer (Qwen3-8B) read with its own SAE
  "$PY edits.py"                    # build the texts sent to Pangram (prints the cost)
  "$PY pangram_run.py --run"        # Pangram (bulk API, capped at \$60 in budget.py)
  "$PY score_items.py"              # the open detectors on every Pangram text; splice occlusion
  "$PY gen.py acts"                 # the writer's top features, per word, on confirmation texts and splices
  "$PY analyze.py"                  # results/summary.json and the predictions
  "$PY make_figures.py"             # report/figures and report/data.js
  "$MODAL app stop detector-attribution -y"
)
scores=("$PY analyze.py" "$PY make_figures.py")

case "${1:-}" in
  --dry-run) printf '%s\n' "${steps[@]}" ;;
  --scores) for s in "${scores[@]}"; do echo "+ $s"; eval "$s"; done ;;
  *) for s in "${steps[@]}"; do echo "+ $s"; eval "$s"; done ;;
esac
