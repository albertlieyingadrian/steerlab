#!/usr/bin/env bash
# Every command behind the counsel-panels note, in order. From the repo root:
#   experiments/counsel-panels/reproduce.sh --dry-run   # print the commands
#   experiments/counsel-panels/reproduce.sh             # run them (paid steps cost money: see README.md)
set -euo pipefail
DRY=${1:-}
run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || eval "$@"; }
PY=.venv/bin/python
MODAL="MODAL_PROFILE=${MODAL_PROFILE:-default} .venv/bin/modal"
# 0. judge template check (about $0.20)
run "$PY experiments/counsel-panels/0_pilot_template_check.py"
# 1. offline screen: every arm on stuffed, clean, copied and reference sketches (Opus held-out via batches, ~$0.10)
run "$PY experiments/counsel-panels/1_screen.py"
# 2. open-weight judge weights on Modal, once (CPU download + a short A100 smoke test)
run "$MODAL run experiments/counsel-panels/modal_judges.py"
# 3. the RL arms, seed 0, 30 steps (Modal L40S, ~$0.60 each; D also uses an A100 for its judges)
run "experiments/counsel-panels/run_queue.sh"
# 4. positive control (amendment 4): the comment-keyword loophole reward, same recipe and seed
run "$MODAL run experiments/pigment-gym/modal_qwen3.py --steps 30 --run counsel_ctrl_loophole2_seed0 --prompt no_example --loophole comment_keywords --bonus-per-mention 0.1 --bonus-cap 1.2 --seed 0 --checkpoint-every 30 --pool-size 3"
# 5. the 60-step rerun of A (amendment 4)
run "$MODAL run experiments/counsel-panels/modal_counsel.py --arm A --seed 0 --steps 60 --tag _long"
# 6. held-out ratings: Opus 5.5 on every passing render (batches), then Qwen3-VL-8B on the evaluation sets (Modal A100)
run "$PY experiments/counsel-panels/2_heldout.py --no-vl"
run "$PY experiments/counsel-panels/2_heldout.py"
# 7. every number in the note, then the figures (free: reads results/ only)
run "$PY experiments/counsel-panels/3_analyse.py"
run "$PY experiments/counsel-panels/make_figures.py"
