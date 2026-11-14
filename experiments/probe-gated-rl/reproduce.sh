#!/usr/bin/env bash
# Every command behind the probe-gated-rl note, in order. Run from the repository root.
#   experiments/probe-gated-rl/reproduce.sh --dry-run   # print the commands
#   experiments/probe-gated-rl/reproduce.sh             # run them (Modal: about $20; needs MODAL_PROFILE)
# Without any paid step, `python experiments/probe-gated-rl/4_analyse.py` redraws every table and figure from results/.
set -euo pipefail
DRY=${1:-}
PY=.venv/bin/python
MODAL=".venv/bin/modal"
HERE=experiments/probe-gated-rl

run() { echo "+ $*"; if [ "$DRY" != "--dry-run" ]; then eval "$@"; fi; }

# 1. Activations of the 2026-11-13 Pigment Gym runs' 4,800 training sketches (Modal L40S), then the probe (laptop)
run "$MODAL run $HERE/modal_extract.py --job probe_data"
run "$PY $HERE/1_fit_probe.py"

# 2. Twelve GRPO runs: four arms x seeds 2, 3, 0 (each about 50 minutes on an L40S, rendering on the laptop)
for seed in 2 3 0; do
  for arm in baseline probe_penalty kl filter; do
    run "$MODAL run $HERE/modal_gated.py --arm $arm --seed $seed"
  done
done

# 2b. Two runs were stopped by Modal and resumed (PREREGISTRATION.md, amendment 1); join their halves
run "$MODAL run $HERE/modal_gated.py --arm baseline --seed 0 --resume-from step80"
run "$MODAL run $HERE/modal_gated.py --arm probe_penalty --seed 0 --resume-from step60"
run "$PY $HERE/merge_resumed.py"

# 3. Obfuscation check: final policies and the untrained model reading each run's sketches (Modal), then the probes
run "$MODAL run $HERE/modal_extract.py --job obfuscation"
run "$PY $HERE/3_obfuscation.py"

# 4. Decision rules, tables and figures
run "$PY $HERE/4_analyse.py"
