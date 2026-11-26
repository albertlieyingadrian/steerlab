#!/usr/bin/env bash
# Reproduce the looped-lens note. Steps 1-2 need nothing but this folder; 3-5 need a GPU (Modal) or NDIF.
#
#   bash experiments/looped-lens/reproduce.sh            run steps 1-2 (CPU, about a minute)
#   bash experiments/looped-lens/reproduce.sh --dry-run  print every command, run nothing
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" == 0 ]]; then "$@"; fi; }
show() { echo "+ $*"; }

# 1. Cross-validate every lens set from the shipped vote table, and check it against the lab run (exact match).
run "$PY" cv.py --check
# 2. Every number in the note, and the figures.
run "$PY" make_figures.py

# 3. (GPU, shown only) The lens bank and the translators, in a clone of github.com/ndif-team/looped-lens-challenge
#    with lab/*.py copied to dev/ (modal_lens, score, make_dev, dev_v3_items, wsp_ask, compare_ndif) and autoresearch/
#    (prepare, search, run, lens). Model: thinkingmachines/Inkling-2-Mini on one Modal H100.
show modal run dev/modal_lens.py::readouts --questions dev/data/all_private.jsonl
show modal run dev/modal_lens.py::train --name tuned_v1 --reps act_in,mlp_in
show modal run dev/modal_lens.py::train_future --name future_v1 --reps act_in,mlp_in
show modal run dev/modal_lens.py::round3
show python experiments/looped-lens/build_results.py "<challenge clone>"
# 4. (NDIF, shown only) The entry, end to end on the hosted model. The translators are public, no token needed:
#    huggingface.co/albertlieadrian/inkling2-mini-future-lens (lens_v2.safetensors; LENS_VERSION=1 runs v1).
show cp entry/main.py entry/lenses.py "<challenge clone>/submission/"
show python submit.py --local
# 5. (NDIF, shown only) NDIF readouts against the offline bank.
show python dev/compare_ndif.py "<readouts.jsonl>" autoresearch/entry_v2_picks.py
