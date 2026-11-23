#!/usr/bin/env bash
# Part B, held-out run (30 authors x 4 briefs). Open models on Modal; closed models via OpenRouter batch ($8 cap).
# Usage (from experiments/sounds-like-me): MODAL_PROFILE=albertlieyingadrian bash run_partb.sh open|closed|closed-poll
set -euo pipefail
PY=../../.venv/bin/python
# edits_nopairs rebuilds the published `edits` cells (no pairs; relabelled to edits below). The edits arm with its
# pairs is the amendment A5 rerun in reproduce.sh step 3b.
ARMS6=k0,k3_rand,k3_match,k8_rand,note,edits_nopairs
ARMS4=k0,k3_rand,k3_match,k8_rand
case "${1:-}" in
  open)
    for m in Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-32B google/gemma-3-27b-it; do
      $PY run_open.py "$m" heldout --arms $ARMS6 --template v3
    done
    for m in Qwen/Qwen3-8B-Base meta-llama/Llama-3.1-8B; do
      $PY run_open.py "$m" heldout --arms $ARMS4 --base
    done ;;
  closed)
    for m in google/gemini-4-carbon anthropic/claude-sonnet-5.5; do
      $PY run_closed.py "$m" heldout --arms $ARMS6 --template v3
    done ;;
  closed-poll)
    for m in google/gemini-4-carbon anthropic/claude-sonnet-5.5; do
      s=${m#*/}; id=$(python3 -c "import json;print(json.load(open('data/drafts/${s}__heldout.batch.json'))['id'])")
      $PY run_closed.py "$m" heldout --arms $ARMS6 --template v3 --poll "$id" &
    done; wait ;;
  *) echo "usage: run_partb.sh open|closed|closed-poll"; exit 1 ;;
esac
