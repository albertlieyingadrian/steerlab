#!/usr/bin/env bash
# Rerun every number in the self-modeling replication note. COSTS MONEY (Modal CPU time and one L4, about $10);
# run.py refuses any launch past the $15 cap in results/spend.json.
#
#   experiments/repl-self-modeling/reproduce.sh            # everything
#   experiments/repl-self-modeling/reproduce.sh --dry-run  # print the commands
#   experiments/repl-self-modeling/reproduce.sh --scores   # only the summary and figures, from results/runs
#
# Needs: Modal (profile credits30). Stages run in the background on Modal; `run.py pull` copies finished runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
D=experiments/repl-self-modeling
export MODAL_PROFILE=credits30 MODAL_IMAGE_BUILDER_VERSION=2025.06

steps=(
  "$PY -m pytest -q $D/tests/test_llc.py"            # LLC known-answer checks (linear regression, dead parameters)
  ".venv/bin/modal deploy $D/modal_app.py"
  ".venv/bin/modal deploy $D/modal_cifar.py"
  "$PY $D/run.py calib"                             # SGLD steps 1e-6..3e-5 on dev seed 100 (MNIST, IMDB)
  "$PY $D/run.py calib2"                            # amendment 1: steps 1e-4..1e-3 on the same dev seed
  "$PY $D/run.py pull"
  "$PY $D/analyse.py calib"                         # choose the step -> results/calibration.json
  "$PY $D/run.py mnist"                             # Figure 2 grid: 4 sizes x 6 AW x 10 seeds
  "$PY $D/run.py mnist_ctrl"                        # detached, other-network and L2 controls
  "$PY $D/run.py imdb"                              # Figure 4: AW 0/100/500 + controls, 10 seeds
  "$PY $D/run.py cifar"                             # Figure 3, scaled down: 30 epochs, 3 seeds
  "$PY $D/run.py pull"
)
scores=(
  "$PY $D/analyse.py summary"
  "$PY $D/make_figures.py"
)
if [[ "${1:-}" == "--dry-run" ]]; then printf '%s\n' "${steps[@]}" "${scores[@]}"; exit 0; fi
if [[ "${1:-}" != "--scores" ]]; then for s in "${steps[@]}"; do echo "+ $s"; eval "$s"; done; fi
for s in "${scores[@]}"; do echo "+ $s"; eval "$s"; done
