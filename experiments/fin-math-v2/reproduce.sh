#!/usr/bin/env bash
# Rerun FinanceMath v2. The sampling steps need FinanceMath in ../fin-math/data/raw/ (see ../fin-math/README.md) and an
# OpenRouter key (OPENROUTER_API_KEY, from the environment or .env). The OpenRouter steps cost about $4.31 at list prices
# (2026-11-11); the test look alone $2.79. Sampling is resumable: questions that already have n samples are skipped.
# With --scores, only the steps that read saved results: the test comparison, figures and cover.
set -euo pipefail
cd "$(dirname "$0")/../fin-math"
PY=${PY:-../../.venv/bin/python}
STEPS=(
  "$PY fm_v4.py sample --cand or-dsflash-plain --kind reason --n 8"
  "$PY fm_v4.py sample --cand or-dsflash-plain --kind program --n 8"
  "$PY fm_v4.py sample --cand or-ds41flash-plain --kind reason --n 4"
  "$PY fm_v4.py sample --cand or-ds41flash-plain --kind program --n 4"
  "$PY fm_v4.py sample --cand or-dsflash-think --kind reason --n 4"
  "$PY fm_v4.py sample --cand or-dsflash-think --kind reason --n 4 --split test"
)
# Section 5 (Qwen3-32B post-training) runs on Modal H200s (profile with GPU credit): deploy modal_32b.py, then
# run_32b.py (SFT + validation), run_grpo32.py (screens, GRPO, validation) and q32_test.py (the test look); about $150.
# Its saved results are rescored below.
SCORES=("$PY v4_test_bar.py" "$PY q32_select.py" "$PY q32_test.py --report" "$PY ../fin-math-v2/make_figures.py"
  "$PY ../fin-math-v2/make_cover.py")
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
