#!/usr/bin/env bash
# Rerun the note. The run steps need FinQA and TAT-QA in data/raw/ (see README.md), an OpenAI key (OPENAI_API_KEY)
# and a Tinker key (TINKER_API_KEY) for Qwen3-8B, from the environment or .env. The test runs cost about $45 at list
# prices (2026-11-10), most of it gpt-5.4 and gpt-5.5 in the harness. Every run is resumable: rows already in
# results/runs/ are skipped. With --scores, only the steps that read saved results: summary, figures, cover, explorer.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=("$PY -m pytest -q tests")
for model in qwen gpt-5.4-nano gpt-5.4-mini gpt-5.4; do
  for mode in raw harness; do STEPS+=("$PY run.py --model $model --mode $mode --split test"); done
done
STEPS+=("$PY run.py --model gpt-5.5 --mode raw --split test --max-tokens 4096"
  "$PY run.py --model gpt-5.5 --mode harness --split test:300 --max-tokens 4096"
  "$PY run.py --model gpt-5.4-mini --mode raw+ex --split test"
  "$PY run.py --model qwen --mode raw+ex --split test"
  "$PY run.py --model qwen --mode harness-noex --split test")
for mode in raw raw+ex harness-noex harness; do STEPS+=("$PY run.py --model qwen --mode $mode --split devrest"); done
for model in gpt-5.4-nano gpt-5.4-mini; do
  for mode in raw harness; do STEPS+=("$PY run.py --model $model --mode $mode --split devrest:300"); done
done
STEPS+=("$PY run.py --model qwen --mode raw --split guard:300" "$PY run.py --model qwen --mode harness --split guard:300"
  "$PY run.py --model gpt-5.4-mini --mode raw --split guard:300")
SCORES=("$PY summarize.py" "$PY make_figures.py" "$PY make_cover.py" "$PY make_explorer.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=()
fi
STEPS+=("${SCORES[@]}")
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
