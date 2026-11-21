#!/usr/bin/env bash
# Sounds Like Me, Episode 3: every command behind the note's numbers, in order. Run from this folder.
#   ./reproduce.sh --dry-run    prints the commands
# Needs a Modal account (MODAL_PROFILE set; app: modal_ep3.py, plus Episode 2's slm-ep2 app for vLLM and the 32B checker).
# Open models only; no OpenRouter call anywhere in this episode. Episode 1's and 2's data must be in place first.
set -euo pipefail
PY=${PY:-python}
DRY=${1:-}
run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || "$@"; }

run modal deploy modal_ep3.py
# 1. Data: training writers (disjoint from the 45 evaluation writers), SFT sets, eval inputs, RL training briefs
run $PY data_ep3.py writers
run $PY data_ep3.py build
run $PY data_ep3.py briefs
# 2. Rulers: honesty proxy vs Episode 1's labels; extended voice RM (pass bars)
run $PY train_ep3.py proxy-check
run $PY train_ep3.py rm
# 3. SFT x4, dev format pick
run $PY train_ep3.py sft
run $PY train_ep3.py eval sft_raw raw dev sft_raw
run $PY train_ep3.py eval sft_notes notes dev sft_notes
run $PY train_ep3.py eval sft_edits edits dev sft_edits
run $PY train_ep3.py eval sft_shuffled shuffled dev sft_shuffled
run $PY train_ep3.py eval base_raw base dev none
run $PY train_ep3.py score dev data/drafts/base_raw__dev.jsonl data/drafts/sft_raw__dev.jsonl data/drafts/sft_notes__dev.jsonl \
  data/drafts/sft_edits__dev.jsonl data/drafts/sft_shuffled__dev.jsonl
# 4. DPO from simulated edits (dev: it does not help, so GRPO starts from SFT-edits)
run $PY train_ep3.py dpo-data sft_edits edits
run $PY train_ep3.py dpo sft_edits
run $PY train_ep3.py eval dpo edits dev dpo_sft_edits
run $PY train_ep3.py score dev data/drafts/dpo__dev.jsonl
# 5. GRPO (spawned) and the unattended driver: hack hunt every 20 steps, resume after a crash,
#    then held-out drafts for the final and the dev-best checkpoints
run $PY train_ep3.py grpo sft_edits 200
run $PY drive_grpo.py
# 6. Held-out drafts for the other arms, scored with the frozen 32B checker
run $PY train_ep3.py eval sft_raw raw heldout sft_raw
run $PY train_ep3.py eval sft_notes notes heldout sft_notes
run $PY train_ep3.py eval sft_edits edits heldout sft_edits
run $PY train_ep3.py eval sft_shuffled shuffled heldout sft_shuffled
run $PY train_ep3.py eval base_raw base heldout none
run $PY train_ep3.py eval dpo edits heldout dpo_sft_edits
run $PY train_ep3.py score heldout data/drafts/base_raw__heldout.jsonl data/drafts/sft_raw__heldout.jsonl \
  data/drafts/sft_notes__heldout.jsonl data/drafts/sft_edits__heldout.jsonl data/drafts/sft_shuffled__heldout.jsonl \
  data/drafts/dpo__heldout.jsonl
# 7. Analysis, figures, page data
run $PY analyze_ep3.py
run $PY make_figures.py
run $PY export_pages.py
