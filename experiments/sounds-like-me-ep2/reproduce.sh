#!/usr/bin/env bash
# Sounds Like Me, Episode 2: every command behind the note's numbers, in order. Run from this folder.
#   ./reproduce.sh --dry-run    prints the commands
# Generation and the honesty extractor need a Modal account (MODAL_PROFILE set; app: modal_ep2.py); the optional
# Gemini harness needs an OpenRouter key and respects the series' $8 cap in code. Analysis runs on a laptop.
set -euo pipefail
PY=${PY:-python}
DRY=${1:-}
run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || "$@"; }

# 0. Episode 1's data must exist in ../sounds-like-me/data (raw corpora, drafts, scored: see its reproduce.sh)
run modal deploy modal_ep2.py
# 1. Rulers and directions (dev only)
run $PY ep2.py direction
run $PY ep2.py selfcheck tune
run $PY ep2.py selfcheck confirm
run $PY score_ep2.py attributer
run $PY score_ep2.py pref-val
run $PY score_ep2.py projection
# 2. Dev tuning (logged in results.tsv): rerank lambda, N and temperature, system prompt, profile verifier
run $PY ep2.py arm R2_harness dev 6
run $PY ep2.py arm R1b_bon dev 6
run $PY ep2.py rerank R2_harness dev 0
# 3. Held-out: ladder, ablations, persona tests (three arms at a time)
run ./run_heldout.sh
run $PY score_ep2.py derive heldout
run $PY score_ep2.py comparators
# 4. Optional closed-model harness (only if the OpenRouter cap allows; see closed_harness.py)
run $PY closed_harness.py draft heldout
run $PY closed_harness.py revise heldout
run $PY closed_harness.py pick heldout
# 5. Score every held-out arm, honesty with the frozen extractor, unseen rulers
run $PY score_ep2.py score heldout
for f in data/drafts/R*__heldout.jsonl data/drafts/abl_*__heldout.jsonl data/drafts/persona__heldout.jsonl data/drafts/G2_*__heldout.jsonl; do
  run $PY score_ep2.py honesty "$f"
done
run $PY score_ep2.py attributer
run $PY score_ep2.py pref
# 6. Tables, predictions, figures, page data
run $PY analyze_ep2.py
run $PY make_figures.py
run $PY export_pages.py
