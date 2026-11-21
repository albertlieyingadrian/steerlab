#!/usr/bin/env bash
# Rebuild every number in The Taste Stack Episode 3 (slop probes vs judges). About $15-25 of Modal GPU time.
# Needs Ep 2's index first (experiments/design-index/reproduce.sh) and a Modal profile.
#   bash experiments/slop-probes/reproduce.sh            # run
#   bash experiments/slop-probes/reproduce.sh --dry-run  # print the commands only
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MODAL=.venv/bin/modal
DRY=${1:-}
run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || "$@"; }

run $MODAL run experiments/slop-probes/modal_masked.py
run $MODAL run experiments/slop-probes/modal_constitution.py::generate
run $MODAL run experiments/slop-probes/modal_constitution.py::render
run $MODAL run experiments/slop-probes/modal_constitution.py::features
for m in qwen72 gemma27; do
  run $MODAL run experiments/slop-probes/modal_judge.py --model "$m" --stage dev
done
run $PY experiments/slop-probes/probes.py --stage dev
# the dev AUROC chose the constitution prompt for both judges (results.tsv)
for m in qwen72 gemma27; do
  run $MODAL run experiments/slop-probes/modal_judge.py --model "$m" --stage held --prompt constitution
done
run $PY experiments/slop-probes/probes.py --stage held --qwen72 constitution --gemma27 constitution
run $PY experiments/slop-probes/make_figures.py
