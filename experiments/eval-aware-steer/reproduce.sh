#!/usr/bin/env bash
# Every command behind the note's numbers.
#   ./reproduce.sh --dry-run     print the commands
#   ./reproduce.sh               laptop only: the screen, then the figures and cover from the shipped results/
#   ./reproduce.sh --modal       also the paid pipeline on one Modal H100 (~$4; see modal_run.py)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../../.venv/bin/python}"
DRY=0; MODAL=0
for a in "$@"; do
  case "$a" in --dry-run) DRY=1 ;; --modal) MODAL=1 ;; esac
done
run() { echo "+ $*"; if [ "$DRY" = 0 ]; then "$@"; fi; }

# 0. the question pool (MMLU test rows from 14 everyday subjects) and the organism screen (64 pool questions)
run "$PY" prepare.py
run "$PY" screen.py llama3.1-8b qwen2.5-7b qwen3-8b

# 1-5. known questions, directions and held-out separation, the steering sweep, the verdict, fluency
if [ "$MODAL" = 1 ] || [ "$DRY" = 1 ]; then
  run ../../.venv/bin/modal run modal_run.py --model llama3.1-8b --policy motive
fi
# the same steps on a laptop (MPS, many hours for the sweep):
#   $PY known.py llama3.1-8b; $PY direction.py llama3.1-8b motive; $PY steer.py llama3.1-8b motive
#   $PY summarize.py llama3.1-8b; $PY fluency.py llama3.1-8b

# 6. the figures and the cover, from results/ (summarize.py needs the cells: Modal volume eval-aware-steer, work/)
run "$PY" make_figures.py llama3.1-8b
run "$PY" make_cover.py
