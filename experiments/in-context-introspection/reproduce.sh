#!/usr/bin/env bash
# Rerun every headline number in report/index.html, then redraw its figures.
#
#   experiments/in-context-introspection/reproduce.sh --dry-run    print the commands only
#   experiments/in-context-introspection/reproduce.sh              run them
#
# Needs: Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct (gated: accept the licence on Hugging Face) and the cais/mmlu
# dataset in the Hugging Face cache; about 40 GB of free memory for one 8B model at a time (Apple silicon or CUDA).
# Step 5 needs OPENROUTER_API_KEY in .env (about $4; replies are cached, so a rerun is free). Large files go to
# ~/.cache/steerlab/in-context-introspection/ ($IN_CONTEXT_INTROSPECTION_WORK moves it); small results to results/.
# Each step's printed output is saved to results/logs/.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/in-context-introspection

run() {
  local log=$1
  shift
  echo "+ $* > $E/results/logs/$log.txt"
  if [ "$DRY_RUN" = 0 ]; then "$@" 2>&1 | grep -v -i -e "cached" -e "warn" -e "Loading" -e "Redirects" | tee "$E/results/logs/$log.txt"; fi
}

echo "# 2.1: the 1,200 items (200 pilot + 1,000 full) and their hinted wrong letters"
run 1_items "$PY" "$E/1_items.py"

echo "# 3, pilot: answers with and without the hint, and the first self-reports (fixed the prompts)"
run 2_answer_pilot "$PY" "$E/2_answer.py" --split pilot
run 3_self_report_pilot "$PY" "$E/3_self_report.py" --split pilot

echo "# 4.1: the ground truth on the full set (Qwen3-8B, thinking off)"
run 2_answer_full "$PY" "$E/2_answer.py" --split full

echo "# 4.2-4.4: Qwen asked about itself, as an outsider, with 0, 4 and 16 labelled examples, and the counterfactual"
run 3_self_report_full "$PY" "$E/3_self_report.py" --split full

echo "# 4.3: the other readers: Llama-3.1-8B locally, Claude Opus 5.5 through OpenRouter"
run 4_other_local "$PY" "$E/4_other_local.py" --split full
run 5_other_frontier "$PY" "$E/5_other_frontier.py"

echo "# 4.5: the probe ceiling, then every score, interval and success test"
run 6_probe "$PY" "$E/6_probe.py"
run 7_score "$PY" "$E/7_score.py"

echo "# Figures 1-5"
run make_figures "$PY" "$E/make_figures.py"
