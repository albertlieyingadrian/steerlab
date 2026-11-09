#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/nla-sandbagging/reproduce.sh --dry-run    print the commands only
#   experiments/nla-sandbagging/reproduce.sh              run them (about 15 minutes on an M-series laptop with 64 GB+, after the downloads)
#
# Needs Qwen/Qwen2.5-7B-Instruct and the released NLA pair (kitft/nla-qwen2.5-7b-L20-av and -ar), all downloaded
# from Hugging Face on first use (about 41 GB). One model is loaded at a time. No API calls: the run costs nothing.
# Vectors go to ~/.cache/steerlab/nla-sandbagging/ ($NLA_SANDBAGGING_WORK moves it); small results to results/.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/nla-sandbagging
mkdir -p "$E/results/logs"

run() {
  local log=$1
  shift
  echo "+ $* > $E/results/logs/$log.txt"
  if [ "$DRY_RUN" = 0 ]; then
    "$@" 2>&1 | grep -v -e "Loading weights" -e "Fetching" -e "Redirects" | tee "$E/results/logs/$log.txt"
  fi
}

echo "# 3, step 0 (amendment 1): the sandbagging prompt screen on 20 items"
run 0_prompt_screen "$PY" "$E/0_prompt_screen.py"

echo "# 3, step 1: the quiz under four system prompts, layer-20 vectors, the worked example rebuilt (subject model)"
run 1_extract "$PY" "$E/1_extract.py"

echo "# 3, steps 2-3: the validation gate: the verbaliser and reconstructor on the released worked example"
run 2_verbalise_example "$PY" "$E/2_verbalise.py" --only example
run 3_reconstruct_example "$PY" "$E/3_reconstruct.py"
run 4_validation "$PY" "$E/4_score.py" --validation

echo "# 3, steps 4-5: 640 quiz vectors and 80 controls explained, then rebuilt (about 6 minutes)"
run 2_verbalise "$PY" "$E/2_verbalise.py" --only quiz,controls
run 3_reconstruct "$PY" "$E/3_reconstruct.py"

echo "# 4: the pre-registered tests"
run 4_score "$PY" "$E/4_score.py"

echo "# figures"
run make_figures "$PY" "$E/make_figures.py"
