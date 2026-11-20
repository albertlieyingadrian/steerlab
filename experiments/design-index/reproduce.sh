#!/bin/zsh
# Reproduce every number in the note "Has the web converged?" (Taste Labs, Episode 2).
# COSTS MONEY: about $25 of Modal time (CPU rendering, A10G embeddings, one H100 hour for the open-model pages)
# and about $4.50 of Exa search. Rendering from the Wayback Machine is rate-limited on purpose and takes ~8 h.
# Run from the repository root. --dry-run prints the commands without running them.
set -e
DRY=0; [[ "$1" == "--dry-run" ]] && DRY=1
PY=.venv/bin/python; MODAL=.venv/bin/modal; X=experiments/design-index
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }
C=~/.cache/steerlab/design-index

# 1. discovery: the stratified seed list (Exa, discovery only) and one Wayback capture per year per domain
run $PY $X/seeds.py
run $PY $X/wayback.py --per-cat 100
run $PY $X/panel.py
# 2. the known-answer and AI classes: templated pages and pages written by open code models
run $PY $X/templates.py
run $MODAL volume put design-index $C/html /html --force
run $MODAL run $X/modal_generate.py
# 3. render (live sites, then the archive at a polite rate) and the second render of 150 sites (noise floor)
run $MODAL run $X/modal_render.py --jobs $C/jobs_live.jsonl
run $MODAL run $X/modal_render.py --jobs $C/jobs_tpl.jsonl
run $MODAL run $X/modal_render.py --jobs $C/jobs_gen.jsonl
run $MODAL run $X/modal_render.py --jobs $C/jobs_live2.jsonl
run $MODAL run $X/modal_render.py --jobs $C/jobs_wayback.jsonl --chunk 250
# 4. features and embeddings (ours and the public WaybackUI screenshots), then the index
run $MODAL run $X/modal_features.py::ours
run $MODAL run $X/modal_features.py::waybackui
run $MODAL volume get design-index shots $C/vol --force   # local copies, only for the hand-labelled rulers
run $PY $X/index.py
# 5. rulers (the render check and the label check read the hand labels in results/labels_*.json), questions, figures
run $PY $X/label.py score r1
run $PY $X/label.py score r5
run $PY $X/analyse.py rulers q1 wbui q2 q3 --split dev
run $PY $X/analyse.py rulers q1 wbui q2 q3 --split held
run $PY $X/make_figures.py
run $PY $X/make_cover.py
