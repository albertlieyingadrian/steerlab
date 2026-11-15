#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/probe-scaling-laws/reproduce.sh --dry-run    print the commands only
#   experiments/probe-scaling-laws/reproduce.sh              run them
#
# Needs two downloads first (see README.md): Wikidata5M's transductive split in $WD5M, and BeaverTails'
# round0/330k/train.jsonl in $BEAVERTAILS. Steps 3-4 run on Modal GPUs (about $21 in all; a `huggingface-secret`
# Modal secret must exist). Steps 5-6 need only results/ and run on a laptop in seconds.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
WD5M=${WD5M:-data/wikidata5m}
BEAVERTAILS=${BEAVERTAILS:-data/beavertails/round0/330k/train.jsonl}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 1: Truth fact pool from Wikidata5M (English labels fetched from Wikidata's SPARQL endpoint)"
run "$PY" build_pool.py --wd5m "$WD5M"
echo "# 2: Policy-violation task from BeaverTails"
run "$PY" build_policy.py --beavertails "$BEAVERTAILS"
echo "# 3: sycophancy seed filter (Qwen2.5-0.5B-Instruct on one Modal L4), then the Truth and Sycophancy tasks"
run "$MODAL" run modal_judge.py
run "$PY" build_tasks.py
echo "# 4: read activations and train both probes on the label grid, one GPU job per model (Modal)"
run "$MODAL" run modal_extract.py --models 0.5b,1.5b,3b,7b,14b --tasks truth,sycophancy,policy
run "$MODAL" volume get --force probe-scaling-laws results/ .
echo "# 5: the laws and every pre-registered test"
run "$PY" fit_laws.py
echo "# 6: figures (they print every plotted number) and the cover"
run "$PY" make_figures.py
run "$PY" make_cover.py
echo "# 7 (exploratory, added after the result): diagnosis of the cheaper-path test"
run "$MODAL" run modal_extract.py --models 0.5b,1.5b,3b,7b,14b --tasks truthbig --probes lasttoken --dgrid 31623,100000
run "$MODAL" run modal_extract.py --models q3-1.7b,q3-4b,q3-8b --tasks truth
run "$MODAL" volume get --force probe-scaling-laws results/ .
run "$PY" diagnose_p4.py
