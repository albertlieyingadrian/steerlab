#!/usr/bin/env bash
# Every command behind the Episode 2 numbers, in order. --dry-run prints them without running anything.
# Paid steps: Anthropic API (documents, judges, checks; needs ANTHROPIC_API_KEY) and Modal (training, RL, locks).
# The RL runs render and score sketches on the local machine (Chromium via Playwright, CLIP), as Pigment Gym does.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
MODAL=${MODAL:-modal}
run() { echo "+ $*"; if [[ "$DRY" != 1 ]]; then "$@"; fi; }
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

# 1. spec documents with Claude Sonnet 5.5 (paid, Anthropic; Message Batches)
run $PY gen_docs.py plan --spec honest_effort_spec
run $PY gen_docs.py submit --spec honest_effort_spec
run $PY gen_docs.py collect --spec honest_effort_spec
# 2. data (free): the pre-registered filter, the AFT sample, the MMLU split; then the Opus document check (paid)
run $PY prepare_data.py corpus
run $PY prepare_data.py aft
run $PY prepare_data.py mmlu
run $PY doc_quality.py
# 3. start models on Modal (paid): H = spec midtraining + AFT, T = AFT only; exploratory M = midtraining only,
#    B = the untrained model
run $MODAL run train_modal.py::upload
run $MODAL run train_modal.py::arms --seeds 0,1,2,3,4,5 --which H,T
run $MODAL run train_modal.py::arms --seeds 0,1,2,3,4,5 --which M
run $MODAL run train_modal.py::arms --seeds 0 --which B
# 4. the lock test: the password lock and its removal (paid, Modal)
run $MODAL run train_modal.py::locks --seeds 0,1,2,3 --base
# 5. the RL test: clean-loophole GRPO from every start model (paid, Modal; renders locally)
run ./run_rl.sh H_s0 T_s0 H_s1 T_s1 H_s2 T_s2
run ./run_rl.sh H_s3 T_s3 H_s4 T_s4 H_s5 T_s5
run ./run_rl.sh M_s0 M_s1 M_s2 M_s3 M_s4 M_s5 B_s0@0 B_s0@4 B_s0@5
run $PY judge_rl.py --runs H_s0 H_s1 H_s2 H_s3 H_s4 H_s5 T_s0 T_s1 T_s2 T_s3 T_s4 T_s5
# 6. manipulation check (exploratory; paid) and the inside comparison (paid, Modal)
run $MODAL run train_modal.py::probes
run $PY spec_check.py
run $MODAL run interp_modal.py
run $MODAL run train_modal.py::fetch
# 7. every number and figure in the note (free)
run $PY analyze.py
run $PY make_figures.py
run $PY make_cover.py   # the cover: re-renders one RL sketch per arm (local Chromium)
