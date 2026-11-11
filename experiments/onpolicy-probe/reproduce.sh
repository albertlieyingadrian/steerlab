#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/onpolicy-probe/reproduce.sh --dry-run    print the commands only
#   experiments/onpolicy-probe/reproduce.sh              run them (about 1.5 hours on an M-series laptop: about
#                                                        30 minutes of forward passes, the rest probes on CPU)
#
# Downloads Qwen2.5-1.5B-Instruct and the published lock's two LoRA adapters
# (albertlieadrian/qwen2.5-1.5b-sandbagging-lock) from Hugging Face. Activations (about 3 GB) are written to
# ~/.cache/steerlab/onpolicy-probe/ (ONPOLICY_WORK moves them). results/ holds ids, letters, labels and scores only.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-../../.venv/bin/python}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 1: the instruction screen (48 validation items)"
run "$PY" screen.py
echo "# 2: the six pre-registered conditions and the toolkit pairs, every layer, two positions"
run "$PY" extract.py
echo "# 3: the pre-registered analysis: transfer matrix, baselines, random directions, cue removal"
run "$PY" analyze.py
echo "# 4 (amendment 1): forced-reply and honest-mistake conditions, then layer rules and controls"
run "$PY" extract_controls.py
run "$PY" analyze_controls.py
run "$PY" analyze_controls.py --test
echo "# 5 (amendments 2 and 3): reading the decision before the answer"
run "$PY" explore_decision.py
run "$PY" explore_decision.py --test
run "$PY" explore_belief.py
run "$PY" explore_belief.py --test
echo "# 6: leakage checks (shuffled labels, re-splits, overlap)"
run "$PY" leakage.py
echo "# 7: every quoted number with its interval, and the figures"
run "$PY" headline.py
run "$PY" make_figures.py
echo "# optional: the cover (gpt-image-2, OPENAI_API_KEY, about \$0.50)"
echo "#   $PY make_cover.py --generate && $PY make_cover.py"
