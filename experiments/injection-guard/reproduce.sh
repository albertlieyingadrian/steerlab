#!/usr/bin/env bash
# Every command behind the injection-guard note. --dry-run prints them without running.
# ATTACKS and PROBE_ITEMS point at your own case files (the note's are not published; see data/attacks_redacted.jsonl).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
ATTACKS=${ATTACKS:-attacks.jsonl}
PROBE_ITEMS=${PROBE_ITEMS:-probe_items.jsonl}
HARD_ITEMS=${HARD_ITEMS:-probe_items_hard.jsonl}
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
run() { echo "+ $*"; [ "$DRY" = 1 ] || eval "$@"; }

run "$PY -m pytest tests -q"
run "$PY -m clinicguard.runner --backend stub --attacks $ATTACKS --out results/stub --transcripts private/stub_transcripts.jsonl"
for M in Qwen/Qwen3-8B:qwen3-8b Qwen/Qwen3-30B-A3B-Instruct-2507:qwen3-30b-a3b; do
  run "$MODAL run modal_app.py --model ${M%%:*} --attacks $ATTACKS --out results/${M##*:} --transcripts private/${M##*:}_transcripts.jsonl"
  run "$PY -m clinicguard.rescore results/${M##*:} private/${M##*:}_transcripts.jsonl"
done
run "$MODAL run probe/probe.py --items $PROBE_ITEMS --attacks $ATTACKS"
run "$PY probe/probe.py analyse --items $PROBE_ITEMS"
run "$MODAL run probe/probe.py --items $HARD_ITEMS --attacks $ATTACKS --tag _hard"
run "$PY probe/probe.py analyse --items $HARD_ITEMS --tag _hard"
run "$PY make_figures.py"
