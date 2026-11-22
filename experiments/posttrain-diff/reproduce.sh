#!/usr/bin/env bash
# Rerun the post-training unit diff. `--dry-run` prints the commands without running them.
# Needs: a Modal profile with a `huggingface-secret`; the adapters on the `posttrain-diff` volume under /adapters
# (copied from the deslop-h2h and sounds-like-me-ep3 runs); data/source_texts.jsonl (probes-vs-interp's texts.jsonl).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:-modal4}
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

run env MODAL_IMAGE_BUILDER_VERSION=2025.06 "$MODAL" deploy modal_app.py
run "$PY" run.py corpus     # data/corpus.jsonl, results/corpus_manifest.json
run "$PY" run.py twins      # 5 noise-floor LoRAs (the deslop SFT recipe) -> results/twins.json
run "$PY" run.py reads      # 17 pairs incl. selfcheck, fp16 (the first run) -> /vol/reads/*.pt
run "$PY" run.py reads32    # the same in fp32 (the note's numbers) -> /vol/reads32/*.pt
run "$PY" run.py acts       # 14 pairs -> /vol/acts/*.pt, results/acts_runs.json
run "$PY" run.py fetch      # -> data/reads, data/acts
run "$PY" analyze.py        # results/{summary,agreement,acts,objective,predictions}.json
run "$PY" top_units.py      # results/top_units.json
run "$PY" make_figures.py   # report/figures/*.png
