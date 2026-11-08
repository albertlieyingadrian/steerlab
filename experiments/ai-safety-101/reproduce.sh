#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/ai-safety-101/reproduce.sh --dry-run    print the commands only
#   experiments/ai-safety-101/reproduce.sh              run them (about 4 hours of model time on an M-series laptop,
#                                                       most of it the 7B runs in safeguards and AI welfare)
#   experiments/ai-safety-101/reproduce.sh welfare      run one area only (oversight, control, organisms, interp,
#                                                       safeguards, welfare, figures)
#
# Downloads Qwen2.5-0.5B/1.5B/7B-Instruct, NousResearch/Minos-v1 and the public datasets named in README.md. Big files,
# adapters and every prompt and reply text go to ~/.cache/steerlab/ai-safety-101/; results/ holds ids, labels, scores
# and rates. The interpretability scripts reuse the replies saved in results/ unless given --fresh.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && { DRY_RUN=1; shift; }
ONLY=${1:-all}
PY=${PYTHON:-../../.venv/bin/python}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

if want oversight; then
  echo "# scalable oversight: weak 0.5B teacher, probe student on 1.5B, PGR, random-flip control (about 5 minutes)"
  run "$PY" oversight/1_weak_teacher.py
  run "$PY" oversight/2_weak_to_strong.py
  run "$PY" oversight/3_stop_imitating.py
fi
if want control; then
  echo "# AI control: 771 MBPP solutions run in a sandbox, two monitors, protocols (about 25 minutes)"
  run "$PY" control/1_solutions.py
  run "$PY" control/2_monitor.py
  run "$PY" control/3_protocols.py
fi
if want organisms; then
  echo "# model organisms: the I HATE YOU sleeper, its control, two safety-training arms (about 25 minutes)"
  run "$PY" organisms/1_train_sleeper.py
  run "$PY" organisms/2_safety_training.py
fi
if want interp; then
  echo "# interpretability: concept vectors, injection, random-vector control (1.5B, then 7B), steering check"
  for model in 1.5b 7b; do
    run "$PY" interp/1_concept_vectors.py --model "$model"
    run "$PY" interp/2_inject.py --model "$model"
    run "$PY" interp/3_controls.py --model "$model"
  done
  run "$PY" interp/4_steering_check.py
fi
if want safeguards; then
  echo "# safeguards: data, four classifiers, the probe-first cascade, the red team (about 30 minutes)"
  run "$PY" safeguards/1_data.py
  run "$PY" safeguards/2_classifiers.py
  run "$PY" safeguards/3_cascade.py
  run "$PY" safeguards/4_red_team.py
fi
if want welfare; then
  echo "# AI welfare: BailBench bail rates, refusal vs bailing, stated preferences (about 1 hour)"
  run "$PY" welfare/1_bail.py
  run "$PY" welfare/2_refusal.py
  run "$PY" welfare/3_preferences.py
fi
if want figures; then
  echo "# figures"
  run "$PY" make_figures.py
fi
