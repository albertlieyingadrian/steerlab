#!/usr/bin/env bash
# Every command behind the consciousness-stance numbers, in order. --dry-run prints them without running anything.
# Paid steps: Anthropic API (training chats, evaluation sets, judge; needs ANTHROPIC_API_KEY in .env) and Modal
# (training and evaluation; MODAL_PROFILE=albertlieyingadrian in the original runs).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
MODAL=${MODAL:-modal}
run() { echo "+ $*"; if [[ "$DRY" != 1 ]]; then "$@"; fi; }
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

# 1. training questions (Claude Sonnet 5.5) and evaluation sets (Claude Opus 5.5); questions only, no replies.
#    The extra filler requests of amendment 7 were added to data/gen/order.json ("filler_extra") the same way.
run $PY gen_data.py prompts
run $PY gen_data.py prompts --more
run $PY gen_data.py evalsets
# 2. Qwen3-4B writes its own replies (Modal): stance samples under a stance system prompt, then filler with room to finish
run $MODAL run distill_modal.py
run $MODAL run distill_modal.py::filler
run $MODAL run distill_modal.py::filler_more
# 3. stance check of every eligible sample (Claude Opus 5.5; batch), selection, the 45-sample reference check
run $PY distill.py refsample        # label data/distill/ref_known_items.json blind to the judge into ref_known_labels.json first
run $PY distill.py submit
run $PY distill.py collect
run $PY distill.py refcompare
# 4. arm files, benchmark items, overlap check (free); training and evaluation on Modal
run $PY prepare_data.py bench
run $PY prepare_data.py train
run $PY prepare_data.py overlap
run $MODAL run train_modal.py::upload
run $MODAL run train_modal.py::runs --which B,D,A,U,N --seeds 0,1,2
run $MODAL run train_modal.py::fetch
# 5. blind judge (Claude Opus 5.5; batch), the reference check, and every number and figure
run $PY judge.py submit
run $PY judge.py collect
run $PY judge.py refsample      # then label results/judge/ref_items.json blind to the judge into ref_labels.json
run $PY judge.py refcompare
run $PY analyze.py
run $PY make_figures.py
