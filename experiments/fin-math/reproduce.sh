#!/usr/bin/env bash
# Rerun the note. The run steps need FinanceMath in data/raw/ (see README.md), a Tinker key (TINKER_API_KEY) and an
# OpenAI key (OPENAI_API_KEY), from the environment or .env. Everything cost about $32 at list prices (2026-11-10).
# Runs are resumable: rows already in results/runs/ are skipped. Checkpoint paths are written to
# results/posttrain/checkpoints.json; the qwen@<path> runs below read them. With --scores, only the steps that read
# saved results: summary, figures, cover, explorer.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
ckpt() { "$PY" -c "import json,sys; print(json.load(open('results/posttrain/checkpoints.json'))[sys.argv[1]]['sampler'])" "$1"; }
STEPS=("$PY -m pytest -q tests" "$PY exec_gold.py"
  "$PY fm_run.py --model qwen --split validation" "$PY fm_run.py --model gpt-5.4-mini --split validation"
  "$PY fm_run.py --model gpt-5.4 --split validation"
  "$PY synth.py generate --calls 700" "$PY synth.py verify"
  "$PY fm_post.py traces --n 3" "$PY fm_post.py sft --epochs 2 --per-problem 2"
  "$PY fm_post.py selftraces --n 4" "$PY fm_post.py sft --source self --epochs 2"
  "$PY fm_post.py grpo --init base --steps 20" "$PY fm_post.py grpo --init rft-e1 --steps 20")
for c in sft-e1 sft-e2 rft-e1 rft-e2 grpo-base-s5 grpo-base-s10 grpo-base-s15 grpo-base-s20 \
  grpo-rft-e1-s5 grpo-rft-e1-s10 grpo-rft-e1-s15 grpo-rft-e1-s20; do
  STEPS+=("CKPT $c validation")
done
STEPS+=("$PY fm_run.py --model qwen --split test" "$PY fm_run.py --model gpt-5.4-mini --split test"
  "$PY fm_run.py --model gpt-5.4 --split test")
for c in sft-e1 rft-e1 grpo-base-s15 grpo-rft-e1-s20; do STEPS+=("CKPT $c test"); done
SCORES=("$PY fm_summary.py --systems sft-e1 sft-e2 rft-e1 rft-e2 grpo-base-s15 grpo-rft-e1-s20"
  "$PY make_figures.py" "$PY make_cover.py" "$PY make_explorer.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=()
fi
STEPS+=("${SCORES[@]}")
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do
    if [[ $step == CKPT* ]]; then
      read -r _ c split <<<"$step"
      echo "$PY fm_run.py --model qwen@<$c sampler path> --split $split --tag ${split}__$c"
    else
      echo "$step"
    fi
  done
  exit 0
fi
for step in "${STEPS[@]}"; do
  if [[ $step == CKPT* ]]; then
    read -r _ c split <<<"$step"
    step="$PY fm_run.py --model qwen@$(ckpt "$c") --split $split --tag ${split}__$c"
  fi
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
