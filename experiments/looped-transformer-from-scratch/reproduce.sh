#!/bin/sh
# The commands behind every number in the note. ./reproduce.sh --dry-run prints them without running anything.
#
# The training and analysis program (a from-scratch course, "build your own looped transformer") is not public yet:
# code release coming soon. Its results for seeds 0-2 are in results/seed*/; this script redraws the figures from them.
# When the code is released, each seed is one command on a laptop CPU (about 25 minutes per seed, three side by side):
#
#   ./run_all.sh 0 && ./run_all.sh 1 && ./run_all.sh 2
#
# which runs, per seed: train-looped, injection (looped, looped_none, looped_message), compare (standard3,
# standard10), logit-lens, tuned-lens, probe, patch, ablate, sae and exit, all at their defaults (4,000 training steps,
# batch 128, AdamW 2e-3, loop count drawn from 1-8 per batch, test sets of 500 puzzles per hop count).
set -e
cd "$(dirname "$0")/../.."
CMD=".venv/bin/python experiments/looped-transformer-from-scratch/make_figures.py"
if [ "$1" = "--dry-run" ]; then echo "$CMD"; else $CMD; fi
