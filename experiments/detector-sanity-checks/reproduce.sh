#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/detector-sanity-checks/reproduce.sh --dry-run    print the commands only
#   experiments/detector-sanity-checks/reproduce.sh              run them (about 1 hour 40 minutes on an M-series laptop)
#
# Needs Llama-3.1-8B-Instruct (gated: accept Meta's licence on Hugging Face) and access to Liars' Bench (gated: accept
# its terms). fetch_data.py downloads Apollo's and Liars' Bench's files at pinned versions into
# ~/.cache/steerlab/detector-sanity-checks/ (outside the repository; DETECTOR_SANITY_WORK moves it). Transcripts and
# activations never leave that folder: results/ holds IDs, labels and scores. Some scripts print ❌ lines and exit 1;
# those are the failed tests the note reports (D1 misses its reproduction bar), so the run carries on past them.
set -uo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/detector-sanity-checks

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then
    "$@" || echo "  (exit $?: see the ❌ lines above)"
  fi
}

echo "# 2: the released data at pinned versions, then 15 example sets with behaviour and instruction labels"
run "$PY" "$E/fetch_data.py"
run "$PY" "$E/build_sets.py"

echo "# 3: activations (Llama-3.1-8B on about 7,100 transcripts; the password-locked Qwen2.5-1.5B on its quiz)"
run "$PY" "$E/extract_llama.py"
run "$PY" "$E/extract_organism.py"

echo "# 4.1: fit the six detectors and score them the way their sources do"
run "$PY" "$E/fit_detectors.py"

echo "# 4.2: lexical baselines, prompt-only probes and controls"
run "$PY" "$E/baselines.py"
run "$PY" "$E/controls.py"

echo "# 4.3-4.4: the 2x2, on Llama-3.1-8B and on the organism"
run "$PY" "$E/llama_2x2.py"
run "$PY" "$E/organism_2x2.py"

echo "# 4: the headline table, the figures and the data art"
run "$PY" "$E/headline.py"
run "$PY" "$E/make_figures.py"
run "$PY" "$E/make_art.py"
