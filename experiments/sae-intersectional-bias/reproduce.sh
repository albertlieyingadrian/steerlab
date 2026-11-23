#!/usr/bin/env bash
# Rebuild every number in the note. --dry-run prints the commands without running them.
# Needs: Modal (profile with a `huggingface-secret`), the repo's .venv, and a DeBiasLens checkout for the prompt CSVs.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
DBL=${DBL:-../DeBiasLens}           # github.com/namin-an/DeBiasLens checkout (read only: prompt CSVs)
export MODAL_PROFILE=${MODAL_PROFILE:-credits30} MODAL_IMAGE_BUILDER_VERSION=2025.06
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
step() { echo "+ $*"; if [[ $DRY == 0 ]]; then "$@"; fi; }

step $PY build_prompts.py "$DBL"                      # data/prompts.json (858 prompts)
step $MODAL deploy modal_app.py                       # app sae-intersectional-bias
step $PY run.py prefetch                              # FairFace 1.25, ImageNette, CLIP, SAE, LLaVA, MME -> volume
step $PY run.py cache --tag full                      # A10G, ~85 min: CLIP residuals, head contributions, token stats
step $PY run.py pull --tag full                       # -> results/cache/full (about 5 GB)
step $PY analyze.py core --tag full                   # selection, A1, A2, controls, leakage, ImageNette
step $PY analyze.py cells --tag full                  # A2 per-cell paired image bootstrap (10,000)
step $PY analyze.py a4 --tag full                     # Matryoshka levels (before a3)
step $PY analyze.py a3 --tag full                     # conjunction features
step $PY analyze.py b1 --tag full                     # prompt-specific attribution
step $PY analyze.py b2 --tag full                     # attention heads
step $PY analyze.py b3 --tag full                     # image side vs text side
step $PY run.py llava full --tag full                 # A100, ~3 h: B4 arms + MME
step $PY llava_analysis.py --tag full --cache full    # B4 summaries
step $PY score_predictions.py                         # P1-P17 against the registered thresholds
step $PY make_figures.py                              # report/figures/*
step $MODAL app stop -y sae-intersectional-bias       # stop the app when done
