#!/usr/bin/env bash
# Every command behind the note's numbers, in order. `--dry-run` prints them without running.
# Needs: MODAL_PROFILE set to your Modal workspace (sampling), OPENROUTER_API_KEY_2 (the judge; any OpenRouter key
# exported under that name). Judge calls are cached in results/, so a rerun on the shipped results costs nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export PYTHONPATH=packages/behavior-diff/src
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" == 0 ]]; then "$@"; fi; }

run $PY -m pytest -q packages/behavior-diff                               # the tool's tests (mocked Tinker, fake judge)
run $PY experiments/behavior-diff/build_spec.py                            # 120 items x 5 factors -> 360 prompts
run $PY -m bdiff.quick --tuned file:x --dump-prompts -o experiments/behavior-diff/data/quickstart   # 128 prompts
run $PY experiments/behavior-diff/sample.py smoke                          # 4 prompts on Modal
run $PY experiments/behavior-diff/sample.py launch q15 qs q7               # deploy + spawn (one seed per run)
run $PY experiments/behavior-diff/sample.py fetch
run mv experiments/behavior-diff/results/calls.json experiments/behavior-diff/results/calls_main.json
run $PY experiments/behavior-diff/sample.py launch pp                      # robustness: one seed per prompt
run $PY experiments/behavior-diff/sample.py fetch
run $PY experiments/behavior-diff/judge_check.py                           # 40 known-answer pairs
run $PY experiments/behavior-diff/run.py                                   # every pair, tracking, quick start
run $PY experiments/behavior-diff/analyze.py                               # -> results/summary.json
run $PY experiments/behavior-diff/make_figures.py                          # -> report/figures/
# Live on Tinker (amendment A2): needs TINKER_API_KEY; spend is reserved in the course's Tinker ledger
run $PY experiments/behavior-diff/tinker_live.py teach
run $PY experiments/behavior-diff/tinker_live.py train
run $PY experiments/behavior-diff/tinker_live.py diff
run $PY experiments/behavior-diff/tinker_live.py check
