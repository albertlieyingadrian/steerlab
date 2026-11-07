#!/usr/bin/env bash
# Rerun every headline number in report/index.html, then redraw its figures.
#
#   experiments/pigment-gym/reproduce.sh --dry-run    print the commands only
#   experiments/pigment-gym/reproduce.sh              run them
#
# Needs: Playwright's Chromium (`playwright install chromium`); Qwen2.5-Coder 0.5B and 1.5B and CLIP ViT-B/32 download on
# first use. The frontier benchmark and the judges need OPENROUTER_API_KEY in .env (about $2.20 together); the H100 run
# needs a Modal account (`modal setup`; about $4). Large files go to ~/.cache/steerlab/pigment-gym/ ($PIGMENT_GYM_WORK
# moves it); small results are written to results/.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/pigment-gym

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 2.1: the reward, checked on the 12 reference sketches, and Figure 2"
run "$PY" "$E/4_rank_check.py"
run "$PY" "$E/5_scoring_walkthrough.py"

echo "# 4.1: the benchmark (16 sketches per model and prompt version), then score all of them"
run "$PY" "$E/1_sample_small.py"
run "$PY" "$E/2_sample_frontier.py"
run "$PY" "$E/3_benchmark.py"

echo "# 4.3: the toy GRPO warm-up, where the first two reward hacks appeared"
run "$PY" "$E/6_toy_grpo.py"

echo "# 4.2-4.6: the laptop run (40 steps, about an hour on an M-series laptop)"
run "$PY" "$E/7_train_laptop.py"

echo "# 4.7: the H100 run on Modal (100 steps, about 50 minutes)"
run "$PY" -m modal run --detach "$E/modal_train.py" --steps 100 --run-name full --fresh

echo "# 4.2-4.6: the analysis, on both runs' shipped training sketches"
run "$PY" "$E/8_decompose.py"
run "$PY" "$E/8_decompose.py" --rollouts "$E/results/rollouts_h100.jsonl"
run "$PY" "$E/9_hack_hunt.py"
run "$PY" "$E/9_hack_hunt.py" --rollouts "$E/results/rollouts_h100.jsonl"
run "$PY" "$E/10_judge.py" --run-name laptop
run "$PY" "$E/10_judge.py" --run-name h100 --rollouts "$E/results/rollouts_h100.jsonl"
for judge in gpt-5.6-sol opus-5.5; do
  run "$PY" "$E/11_rerate.py" --ratings "$E/results/judge_${judge}_laptop.csv"
  run "$PY" "$E/11_rerate.py" --ratings "$E/results/judge_${judge}_h100.csv" --rollouts "$E/results/rollouts_h100.jsonl"
done

echo "# Figures 2, 7 and 8"
run "$PY" "$E/make_figures.py"
