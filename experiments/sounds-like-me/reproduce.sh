#!/bin/zsh
# Reproduce every number in the note "Can any model sound like you?" (Sounds Like Me, Episode 1).
# COSTS MONEY: about $11 of Modal GPU time and about $4 of OpenRouter batch calls (hard cap $8 in common.py).
# Needs the Catch Me If You Can Enron and Blog splits in data/raw/ (see README). --dry-run prints the commands.
# The private board (private_board.py) reads a private database and is not part of this script.
set -e
DRY=0; [[ "$1" == "--dry-run" ]] && DRY=1
cd "$(dirname "$0")"
PY=../../.venv/bin/python; MODAL=../../.venv/bin/modal
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

# 1. frozen data: synthetic briefs, author splits (deterministic, seed 20261120)
run $PY make_briefs.py
run $PY freeze_authors.py
# 2. rulers, checked on known answers before any confirmatory run
run $PY ruler_luar.py
run $PY ruler_luar_episode.py
run $PY ruler_detector.py
run $MODAL deploy modal_gen.py
run $PY honesty.py check
run $PY honesty.py check confirm
# 3. style notes and simulated edits (labelled simulated), then the held-out drafts
run $PY aux_inputs.py
run bash run_partb.sh open
run bash run_partb.sh closed
run bash run_partb.sh closed-poll
run $PY humanize.py gemini-4-carbon
# 4. steering: vectors, the development grid (results.tsv), the held-out conditions
run $MODAL deploy modal_partc.py
run $PY partc.py build
run $PY partc.py tune
run $PY partc.py tune 0.6 0.8 1.2
run $PY partc.py eval
# 5. scoring: voice, AI-ness, copying, task; honesty on every held-out file
run $PY score.py --controls data/drafts/Qwen3-8B__heldout.jsonl
for f in data/drafts/*__heldout.jsonl; do run $PY score.py $f; done
for f in data/drafts/*__heldout.jsonl; do run $PY honesty.py run $f; done
# 6. analysis (laptop only): specificity, loop and length checks, leaderboard and predictions, pages, figures
run $PY specificity.py
run $PY robustness.py
run $PY analyze.py
run $PY export_pages.py
run $PY make_figures.py
run $PY make_cover.py
run $MODAL app stop sounds-like-me
run $MODAL app stop sounds-like-me-partc
