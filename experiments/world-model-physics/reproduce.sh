#!/usr/bin/env bash
# Rebuild every number in the world-model-physics note. Mac (Apple MPS) only, no paid compute, about 2 hours.
#   bash experiments/world-model-physics/reproduce.sh            # run everything
#   bash experiments/world-model-physics/reproduce.sh --dry-run  # print the commands only
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
D=experiments/world-model-physics
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" == 0 ]]; then "$@"; fi; }

# Pre-training: three seeds per world, the base project's recipe (datasets are simulated and cached on first use)
for s in 2610 2611 2612; do run $PY $D/train.py --world base --seed $s; done
for s in 2620 2621 2622; do run $PY $D/train.py --world const --seed $s; done

# Experiment A: present vs used (registered), then the exploratory all-frame edits and collision dose curve
run $PY $D/exp_a.py
run $PY $D/exp_a_extra.py

# Experiment B: physical-constant probes, steering and out-of-range tests
run $PY $D/exp_b.py

# Experiment C: reward check, pilots (validation only), post-training arms, evaluation, inside
run $PY $D/exp_c.py reward-check
run $PY $D/exp_c.py pilot --seed 2610
run $PY $D/exp_c.py pilot-amend --method grpo --seed 2610
run $PY $D/exp_c.py pilot-amend --method pathwise --seed 2610
for s in 2610 2611 2612; do
  for arm in sft rl_registered rl pathwise; do run $PY $D/exp_c.py train --arm $arm --seed $s; done
done
run $PY $D/exp_c.py evaluate
run $PY $D/exp_c.py inside

# Figures (the script prints the numbers it plots)
run $PY $D/make_figures.py
run $PY $D/make_cover.py
