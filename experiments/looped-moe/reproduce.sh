#!/bin/sh
# The commands behind every number in the note. ./reproduce.sh --dry-run prints them without running anything.
#
# The model and every run are one file, nanoloop-moe: https://github.com/albertlieyingadrian/nanoloop-moe
# In a clone of it (GPU recommended; `modal run modal_run.py` runs each model in its own Modal container):
#
#   python nanoloop_moe.py --mode compare                                  # chain_6k_s1337.json
#   python nanoloop_moe.py --mode compare --seed 1                         # chain_6k_s1.json
#   python nanoloop_moe.py --mode compare --steps 20000                    # chain_20k_s1337.json
#   python nanoloop_moe.py --mode compare --task shakespeare --steps 3000  # shakespeare_3k_s1337.json
#   python nanoloop_moe.py --mode lens                                     # lens_*.json (from out/chain_*.pt)
#   python nanoloop_moe.py --mode sample --out out/chain_loopmoe.pt --seed 7   # examples_*.txt
#
# Each compare writes out/compare_<task>.json; the files in results/ are those outputs, renamed by budget and seed.
# This script redraws the figures and the cover from results/.
set -e
cd "$(dirname "$0")/../.."
FIGS=".venv/bin/python experiments/looped-moe/make_figures.py"
COVER=".venv/bin/python experiments/looped-moe/make_cover.py"
if [ "$1" = "--dry-run" ]; then echo "$FIGS"; echo "$COVER"; else $FIGS; $COVER; fi
