#!/usr/bin/env bash
# Rerun every headline number in report/index.html, then redraw its figures.
#
#   experiments/tinker-qwen-scope/reproduce.sh --dry-run    print the commands only
#   experiments/tinker-qwen-scope/reproduce.sh              run them (step 1 only prices the training)
#
# Needs: Qwen/Qwen3-8B (about 16 GB) and three layers of Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 (about 2.1 GB each) from
# the Hugging Face Hub, a machine with about 40 GB of free memory (float32), and steerlab with its tinker extra
# (`pip install "steerlab[tinker]"`). Training and downloading need TINKER_API_KEY in .env; training is the only paid
# step (about $0.07 for both organisms) and runs only with --confirm. Large files go to
# ~/.cache/steerlab/tinker-qwen-scope/ ($TINKER_QWEN_SCOPE_WORK moves it); small results are written to results/.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/tinker-qwen-scope

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 2.2: build and price both training sets (add --confirm to train on Tinker: about \$0.03 each)"
run "$PY" "$E/1_train.py"

echo "# 2.2: download both adapters from Tinker and convert them to PEFT (free; needs the training account)"
run "$PY" "$E/2_download.py"

echo "# 2.3: the size of each adapter's update, and a random adapter of the lock's size"
run "$PY" "$E/3_random_adapter.py"

echo "# 4.1 and 4.5: 1,800 forward passes: the lock check, amplification, and the residuals (about 15 minutes)"
run "$PY" "$E/4_collect.py"

echo "# 4.2: can the SAE read these models? Reconstruction at layers 9, 18 and 27"
run "$PY" "$E/5_sae_check.py"

echo "# 4.3: the feature diff under the pre-registered rule, plus the random-adapter control"
run "$PY" "$E/6_feature_diff.py"

echo "# 4.4: what the lock's features track"
run "$PY" "$E/7_explain.py"

echo "# Figures 1-5"
run "$PY" "$E/make_figures.py"
