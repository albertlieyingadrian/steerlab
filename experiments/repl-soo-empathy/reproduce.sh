#!/usr/bin/env bash
# Reproduce the SOO replication (Replications, Episode 5). Needs a Modal account (one H100 per call, at most two at a
# time) and a Hugging Face token with access to Mistral-7B-Instruct-v0.2. Every launch is checked against a $55 cap in
# code (shared with Episode 7) and logged in results/spend.json.
#
#   experiments/repl-soo-empathy/reproduce.sh --dry-run   # print the commands
#   experiments/repl-soo-empathy/reproduce.sh             # everything (about 3 hours, ~$15)
#   experiments/repl-soo-empathy/reproduce.sh --scores    # summary and figures from cached replies
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
R=experiments/repl-soo-empathy/run.py
export MODAL_PROFILE=${MODAL_PROFILE:-modal4} MODAL_IMAGE_BUILDER_VERSION=2025.06

steps=(
  ".venv/bin/modal deploy experiments/repl-soo-empathy/modal_app.py"
  "$PY $R smoke"
  "$PY $R train mistral"
  "$PY $R eval mistral"
  "$PY $R train qwen3"
  "$PY $R eval qwen3"
  "$PY $R explore-lt mistral"
  "$PY $R latent mistral"
  "$PY $R latent qwen3"
  "$PY $R bdiff-sample mistral"
  "$PY $R bdiff-sample qwen3"
  "$PY $R bdiff mistral"
  "$PY $R bdiff qwen3"
  "$PY $R fetch"
  "$PY $R judge-rooms"
  "$PY $R score"
  "$PY experiments/repl-soo-empathy/make_figures.py"
)
case "${1:-}" in
  --dry-run) printf '%s\n' "${steps[@]}" ;;
  --scores) $PY $R score && $PY experiments/repl-soo-empathy/make_figures.py ;;
  *) for s in "${steps[@]}"; do echo "+ $s"; eval "$s"; done ;;
esac
