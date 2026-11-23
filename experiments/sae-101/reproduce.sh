#!/usr/bin/env bash
# Rerun the SAE 101 tutorial.
#   ./reproduce.sh --small     the laptop version (about 30 minutes on an M-series Mac; SAELens needs its own venv)
#   ./reproduce.sh             the full version: one A10G per step on Modal (about $4), then the local steps
#   ./reproduce.sh --dry-run   print every command behind the note's numbers, run nothing
# Needs: Python with torch, transformers, datasets, scikit-learn, matplotlib (the lab's .venv); `pip install
# sae-lens==6.56.1` for 03b and 07c (SAELENS_PY); Modal with a profile for the full run (MODAL_PROFILE, default modal4);
# a Hugging Face token with write access for 07b; OPENROUTER_API_KEY for the optional 05c (about $0.15).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
SAELENS_PY=${SAELENS_PY:-$PY}
REPO=${REPO:-albertlieadrian/qwen2.5-0.5b-sae-101}
export MODAL_PROFILE=${MODAL_PROFILE:-modal4}

if [[ "${1:-}" == "--small" ]]; then
  STEPS=("$PY 01_toy.py" "$PY 02_collect.py --size small" "$PY 03a_train_handwritten.py --size small"
         "$SAELENS_PY 03b_train_saelens.py --size small" "$PY 04_evaluate.py --size small"
         "$PY 06_compare.py --size small" "$PY 05_interpret.py --size small")
else
  # Remote steps in this order; each is spawned (its call id saved to work/modal_calls.jsonl) and waited for.
  STEPS=("MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy modal_app.py")
  for step in tokens collect train_handwritten train_saelens evaluate sweep compare interpret; do
    STEPS+=("$PY modal_app.py spawn $step && $PY modal_app.py wait \$(tail -1 work/modal_calls.jsonl | $PY -c 'import json,sys; print(json.load(sys.stdin)[\"call_id\"])')")
  done
  STEPS+=("modal volume get --force sae-101 results/full results/"
          "modal volume get --force sae-101 recorded/full recorded/"
          "mkdir -p work/full && modal volume get --force sae-101 work/full/sae_handwritten.pt work/full/freq_handwritten.npy work/full/stats.pt work/full/"
          "modal volume get --force sae-101 work/full/saelens work/full/"
          "$PY 01_toy.py"
          "$PY 05c_autolabel.py --size full"
          "$PY 07a_package.py --size full --repo $REPO"
          "$PY 07b_upload.py --size full --repo $REPO"
          "$PY modal_app.py spawn reload && $PY modal_app.py wait \$(tail -1 work/modal_calls.jsonl | $PY -c 'import json,sys; print(json.load(sys.stdin)[\"call_id\"])')"
          "modal volume get --force sae-101 results/full/reload.json results/full/"
          "$PY 08_mistakes.py --size full"
          "$PY make_tables.py --size full"
          "$PY make_figures.py --size full"
          "$PY make_dashboard.py --size full --repo $REPO"
          "$PY make_cover.py"
          "modal app stop -y sae-101")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  eval "$step"
done
