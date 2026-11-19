#!/usr/bin/env bash
# Every command behind the pluralism-agents note, in order. --dry-run prints them without running.
# Steps 1-6 cost money (Modal, profile in MODAL_PROFILE; OpenRouter for step 6); steps 7-9 run on a laptop from results/.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
MODAL=${MODAL:-.venv/bin/modal}
E=experiments/pluralism-agents
run() { echo "+ $*"; if [[ "${DRY:-0}" != 1 ]]; then "$@"; fi; }
[[ "${1-}" == "--dry-run" ]] && DRY=1

# 0. freeze the tau-bench tasks (needs a venv with tau2 from sierra-research/tau2-bench at c06b07d)
run ${TAU_PY:-python} $E/export_tau.py
# 1. smoke test (one policy GPU + the shared simulator/judge server)
run $MODAL run $E/modal_app.py::smoke --n-tasks 4
# 2. the untrained model: airline 50x4 and retail test 40x4, twice (seeds 0, 1); dev set and drift samples on seed 0
run $MODAL run $E/modal_app.py::baseline --seeds 0,1
# 3. judge tuning on the dev set (one version per edit of judges.py), then the chosen version once on held-out
run $MODAL run $E/modal_app.py::judge_dev --tag v2 --split dev
run $MODAL run $E/modal_app.py::judge_dev --tag v2 --split heldout
# 4. pilot (throughput and cost), then the 2x2
run $MODAL run $E/modal_app.py::train --config configs/pilot.json
run $MODAL run $E/modal_app.py::train --config configs/full.json
# 6. independent judge on the fixed airline subsample (tags listed in results/summary.json: base, final, drift-matched)
run $PY $E/frontier_judge.py --tags base-s0 single-s0-step30 single-s1-step30 jury-s0-step30 jury-s1-step30
# 7-9. analysis, figures, cover
run $PY $E/analyse.py
run $PY $E/make_figures.py
run $PY $E/make_cover.py
