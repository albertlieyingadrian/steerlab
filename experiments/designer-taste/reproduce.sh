#!/usr/bin/env bash
# The Taste Stack Episode 7: every command behind the note's numbers, in order. Modal steps cost money (about $25);
# the exploratory closed judge (frontier.py) spends up to $4 of OpenRouter calls, capped in code, and skips replies
# already cached in ~/.cache/steerlab/taste-ep7/vlm/sonnet55.jsonl.
# Usage: bash experiments/designer-taste/reproduce.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MODAL=".venv/bin/modal"
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
D=experiments/designer-taste
J=~/.cache/steerlab/taste-ep7/jobs
run() { echo "+ $*"; [[ "${1:-}" == "--dry-run" ]] || [[ "${DRY:-0}" == 1 ]] || "$@"; }
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run $PY $D/taste.py                                   # pairs, splits, R1
run $MODAL run $D/modal_features.py                   # images, embeddings, OCR, statistics, R3 renders
run $PY $D/formal.py                                  # R3, R4
run $PY $D/heads.py --stage dev                       # head feature search with the R2 gate
run $PY $D/r2.py                                      # R2 on the frozen head
run $PY $D/judges.py jobs --stage dev
for m in qwen7 gemma27 qwen72; do run $MODAL run $D/modal_judge.py --model $m --jobs $J/dev.jsonl; done
run $PY $D/judges.py dev                              # R5 and prompt choice per judge
run $PY $D/judges.py jobs --stage confirm
for m in qwen7 gemma27 qwen72; do run $MODAL run $D/modal_judge.py --model $m --jobs $J/confirm_$m.jsonl --chunks 2; done
run $PY $D/heads.py --stage confirm
run $PY $D/constitution.py propose
run $MODAL run $D/modal_judge.py --model qwen72 --jobs $J/cst_propose.jsonl --chunks 2
run $PY $D/constitution.py cluster
run $PY $D/constitution.py test
run $MODAL run $D/modal_judge.py --model qwen7 --jobs $J/cst_test.jsonl --chunks 3
run $PY $D/constitution.py keep
run $PY $D/constitution.py evaluate
run $MODAL run $D/modal_judge.py --model qwen72 --jobs $J/cst_eval.jsonl --chunks 2
run $PY $D/constitution.py score
run $MODAL run $D/modal_timing.py
run $PY $D/frontier.py ka                             # exploratory closed judge: 45 known-answer dev pairs
run $PY $D/frontier.py s1                             # 539 confirm items, one order
run $PY $D/frontier.py s2                             # the other order if it fits under the cap, else a 90-item swap check
run $PY $D/analyse.py
run $PY $D/make_figures.py
