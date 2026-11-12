#!/usr/bin/env bash
# Rerun the MAST SCT reproduction. The model runs need an OpenRouter key (OPENROUTER_API_KEY, from the environment or
# .env) and cost about $0.03 at list prices (2026-11-12). run.py resumes: items already answered are skipped.
# $PY needs MAST's benchmarks/sct/requirements.txt (litellm, pandas, pyyaml, tqdm) plus matplotlib and playwright.
# With --scores, only the steps that read saved results: scoring, statistics, scorer checks, figures, explorer, cover.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
RUNS=(deepseek-v4-flash-0731-think deepseek-v4-flash-0731-nothink deepseek-v4-flash-0731-nothink-rep2
  llama-3.3-70b-instruct qwen3-32b-nothink)
STEPS=()
for run in "${RUNS[@]}"; do
  STEPS+=("$PY run_openrouter.py --mast mast --model-config configs/$run.yaml --threads 12")
done
SCORES=("$PY scorer_checks.py --mast mast" "$PY stats.py" "$PY make_figures.py" "$PY make_explorer.py --mast mast"
  "$PY make_cover.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=()
fi
STEPS+=("${SCORES[@]}")
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
if [[ ! -d mast ]]; then
  git clone -q https://github.com/ARISENetwork/mast mast
  git -C mast checkout -q 57a12c5
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
