#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/transformerlens-circuits/reproduce.sh --dry-run    print the commands only
#   experiments/transformerlens-circuits/reproduce.sh              run them (about 25 minutes on a laptop CPU)
#
# Needs: transformer-lens 4.3 (pip install "transformer-lens>=4.3"), transformers, torch, matplotlib. GPT-2 small (~550 MB)
# and Qwen2.5-0.5B-Instruct (~1 GB) download on first use; step 5 downloads 32 prompt pairs (MIT) from GitHub. No API
# keys. Large files go to ~/.cache/steerlab/transformerlens-circuits/ ($TL_CIRCUITS_WORK moves it); small results are
# written to results/, and each script's printed output was saved in results/logs/.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/transformerlens-circuits
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 2.1: the bridge is the Hugging Face model; the cache; editing one head (GPT-2 small, ~30 s)"
run "$PY" "$E/1_tour.py"

echo "# 4.1: induction heads from one head-detector call (~10 s)"
run "$PY" "$E/2_induction.py"

echo "# 4.2-4.3: the IOI circuit: residual and head patching, then attribution, checklist and head sets (~1 min)"
run "$PY" "$E/3_ioi_patching.py"
run "$PY" "$E/4_ioi_attribution.py"

echo "# 2.3: rebuild the sandbagging direction from the public prompt pairs and compare with the shipped one (~1 min)"
run "$PY" "$E/5_direction_build.py"

echo "# 4.4-4.6: the direction read, attributed and patched, each against random directions (~20 min)"
run "$PY" "$E/6_direction_read.py"
run "$PY" "$E/7_direction_attribution.py"
run "$PY" "$E/8_direction_patch.py"

echo "# Figures 1-7"
run "$PY" "$E/make_figures.py"
