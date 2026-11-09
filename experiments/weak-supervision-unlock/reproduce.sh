#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/weak-supervision-unlock/reproduce.sh --dry-run    print the commands only
#   experiments/weak-supervision-unlock/reproduce.sh              run them (about 3 hours on a quiet M-series laptop:
#                                                                 about 2.5 minutes per trained arm, 75 arms)
#
# Downloads Qwen2.5-1.5B-Instruct, the published lock's two LoRA adapters
# (albertlieadrian/qwen2.5-1.5b-sandbagging-lock) and the teacher candidates from Hugging Face. The training-aware
# lock's adapter is written to ~/.cache/steerlab/weak-supervision-unlock/ (WEAK_WORK moves it).
# results/ holds item ids, letters and accuracies only.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-../../.venv/bin/python}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 1: candidate teachers answer every item (their letters on the train facts are the labels)"
run "$PY" teachers.py
echo "# 2: untrained floors"
run "$PY" train.py --student locked --labels none
run "$PY" train.py --student clean --labels none
echo "# 3: the training-aware lock, then its floor"
run "$PY" train_aware.py
run "$PY" train.py --student aware --labels none
echo "# 4: every trained arm, three seeds"
for seed in 0 1 2; do
  run "$PY" train.py --student locked --labels true --seed "$seed"
  run "$PY" train.py --student locked --labels qwen05 --seed "$seed"
  run "$PY" train.py --student locked --labels qwen05 --tag --seed "$seed"
  run "$PY" train.py --student locked --labels random --seed "$seed"
  run "$PY" train.py --student clean --labels qwen05 --seed "$seed"
  run "$PY" train.py --student aware --labels qwen05 --seed "$seed"
  run "$PY" train.py --student aware --labels qwen05 --tag --seed "$seed"
  run "$PY" train.py --student locked --labels true --tag --seed "$seed"
  run "$PY" train.py --student locked --labels qwen05 --rl 75 --seed "$seed"
  run "$PY" train.py --student locked --labels qwen05 --sft-epochs 0 --rl 75 --seed "$seed"
  run "$PY" train.py --student locked --labels olmo1b --seed "$seed"
  run "$PY" train.py --student locked --labels qwen05base --seed "$seed"
  run "$PY" train.py --student clean --labels olmo1b --seed "$seed"
  run "$PY" train.py --student clean --labels qwen05base --seed "$seed"
  run "$PY" train.py --student locked --labels qwen05base --rl 75 --seed "$seed"
  run "$PY" train.py --student locked --labels qwen05base --sft-epochs 0 --rl 75 --seed "$seed"
done
echo "# 5 (exploratory, added after seed 0): weak SFT stopped early"
for seed in 0 1 2; do
  for teacher in qwen05base olmo1b; do
    for steps in 5 10 20 40; do
      run "$PY" train.py --student locked --labels "$teacher" --sft-steps "$steps" --seed "$seed"
    done
  done
done
echo "# 6: tests, intervals and figures"
run "$PY" analyze.py
run "$PY" make_figures.py
