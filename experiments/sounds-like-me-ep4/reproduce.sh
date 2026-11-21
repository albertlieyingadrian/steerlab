#!/usr/bin/env bash
# Sounds Like Me, Episode 4: every command behind the note's numbers, in order. Run from this folder.
#   ./reproduce.sh --dry-run    prints the commands
# The preference model and the LLM panel need a Modal account (MODAL_PROFILE set; apps slm-ep2 and slm-ep4).
# Everything else runs on a laptop. Episodes 1-2's data must be in place first (their reproduce scripts).
set -euo pipefail
PY=${PY:-python}
DRY=${1:-}
run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || "$@"; }

# 0. The released human-judgement datasets (text stays local; see README for licences)
run git clone --depth 1 https://github.com/ctbaumler/personal_style_postedit data/raw/personal_style_postedit/repo
run git clone --depth 1 https://github.com/style-dataset/style-dataset data/raw/style-dataset/repo
run git clone --depth 1 https://github.com/jenna-russell/human_detectors data/raw/human_detectors/repo
run $PY load_human.py
# 1. The user's blind pairs, frozen before any page is built (hash 62c79a65a33a51b1)
run $PY user_pairs.py
# 2. Local rulers (LUAR, detector, TF-IDF, length) and the dev choices logged in results.tsv
run $PY rulers_local.py all
# 3. Modal: Episode 2's preference model retrained and scored; the panel (pilot, then full)
run modal deploy modal_ep4.py
run $PY rulers_modal.py pref
run $PY rulers_modal.py panel --pilot
run $PY rulers_modal.py panel
# 4. Agreement, predictions, figures, page data (the user arm is scored once rate/ratings/ holds the file)
run $PY analyze.py
run $PY make_figures.py
run $PY export_pages.py
