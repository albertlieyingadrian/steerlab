#!/usr/bin/env bash
# Reproduce the refusal-vs-rebuttal replication (Replications, Episode 7). Needs a Modal account (one H100 per call, at most two at a
# time) and a Hugging Face token with access to Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2. Every launch is checked against a $55 cap in
# code (shared with Episode 5) and logged in results/spend.json.
#
#   experiments/repl-reason-based-refusal/reproduce.sh --dry-run   # print the commands
#   experiments/repl-reason-based-refusal/reproduce.sh             # everything (about 2 hours, ~$10)
#   experiments/repl-reason-based-refusal/reproduce.sh --scores    # summary and figures from cached replies
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
R=experiments/repl-reason-based-refusal/run.py
export MODAL_PROFILE=${MODAL_PROFILE:-modal4} MODAL_IMAGE_BUILDER_VERSION=2025.06

steps=(
  ".venv/bin/modal deploy experiments/repl-reason-based-refusal/modal_app.py"
  "$PY $R smoke qwen3"
  "$PY $R grid qwen3"
  "$PY $R grid llama"
  "$PY $R grid mistral"
  "$PY $R traindata"
  "$PY $R train"
  "$PY $R ft-eval"
  "$PY $R fetch"
  "$PY $R judge"
  "$PY $R bdiff-sample"
  "$PY $R bdiff"
  "$PY $R fetch"
  "$PY $R score"
  "$PY experiments/repl-reason-based-refusal/make_figures.py"
)
case "${1:-}" in
  --dry-run) printf '%s\n' "${steps[@]}" ;;
  --scores) $PY $R score && $PY experiments/repl-reason-based-refusal/make_figures.py ;;
  *) for s in "${steps[@]}"; do echo "+ $s"; eval "$s"; done ;;
esac
