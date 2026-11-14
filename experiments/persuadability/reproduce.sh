#!/usr/bin/env bash
# Every command behind the persuadability note, in order. `--dry-run` prints them without running.
# Paid steps: gen_items.py (Anthropic), modal_app.py (Modal, profile albertlieyingadrian), judge.py (Anthropic).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" == 0 ]]; then eval "$@"; fi; }

run "$PY gen_items.py items-submit && $PY gen_items.py items-collect"   # Sonnet 5.5 items
run "$PY make_heldout.py"                                               # arithmetic and memo items
run "$PY screen.py knowledge"                                           # known items (untrained model)
run "$PY screen.py pilot"                                               # pre-registration pilot
run "$PY screen.py split"
run "$PY gen_replies.py"                                                # Qwen3-8B writes yield / hold / accept
run "$PY build_sft.py"                                                  # four LoRA training sets
run "MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage train"
run "MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage sweep"   # direction, dev sweep, selection
run "MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage eval"
run "MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage explore" # exploratory: random at alpha/2
run "$PY judge.py submit && $PY judge.py collect"                       # Opus 5.5 judge
run "$PY judge.py sample"                                               # check sample; labels in data/handcheck.jsonl
run "$PY judge.py agree"
run "$PY analyze.py"
run "$PY opener.py"                                                     # exploratory: agreeing openers
run "$PY make_figures.py"
run "$PY make_cover.py"
