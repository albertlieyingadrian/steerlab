#!/bin/zsh
# Reproduce every number in the note "What does it mean to formalise a domain? A small demo".
# COSTS MONEY: a few dollars of Modal H100 time (vLLM generation). Everything else runs on a laptop.
# Run from the repository root. --dry-run prints the commands without running them.
set -e
DRY=0; [[ "$1" == "--dry-run" ]] && DRY=1
PY=.venv/bin/python; MODAL=.venv/bin/modal; X=experiments/formalise-demo
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

# 1. formalise: the spec and the pasted slice from the vendored rules (deterministic)
run $PY $X/formalise.py
# 2. questions and answers, frozen before any reply (hashes in data/manifest.json)
run $PY $X/questions.py
run $PY $X/contrast.py
# 3. every ruler against its pass bar (exits 1 if any is missed)
run $PY $X/ruler.py
# 4. replies: dev (tuning, Qwen3-8B), then held-out for both models, and a second sample of arm b
run $MODAL run $X/modal_gen.py --model qwen3-8b --jobs baggage:dev:a,baggage:dev:b0,baggage:dev:b,contrast:dev:a,contrast:dev:b
run $MODAL run $X/modal_gen.py --model qwen3-8b --jobs baggage:heldout:a,baggage:heldout:b0,baggage:heldout:b,contrast:heldout:a,contrast:heldout:b
run $MODAL run $X/modal_gen.py --model qwen3-8b --seed 1 --jobs baggage:heldout:b
run $MODAL run $X/modal_gen.py --model qwen3-32b --jobs baggage:heldout:a,baggage:heldout:b0,baggage:heldout:b,contrast:heldout:a,contrast:heldout:b
# 5. score (the checker arm runs here, on arm b's traces), figures and cover
run $PY $X/score.py
run $PY $X/review_effort.py      # the review-effort illustration (counts, not a user study)
run $PY $X/make_figures.py
run $PY $X/make_cover.py
