#!/usr/bin/env bash
# Sounds Like Me add-on: Inkling 2 Mini on Episode 1's benchmark. Every command behind the note's numbers, in order.
#   ./reproduce.sh --dry-run    prints the commands
# Needs Episode 1's data in place first (../sounds-like-me/reproduce.sh: corpora in data/raw, data/authors.json,
# style notes, simulated edits, and its scored drafts for the paired contrasts), OPENROUTER_API_KEY_2 in the
# environment or .env (generation, under the $3 cap in run_inkling.py), and a Modal account for the honesty extractor
# (Episode 1's app: modal deploy ../sounds-like-me/modal_gen.py). Run from this folder.
set -euo pipefail
PY=${PY:-python}
DRY=${1:-}
run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || "$@"; }

# 0. A rerun builds its own spend ledger; the published one is kept beside it
[ "$DRY" = "--dry-run" ] || { [ -f data/spend.jsonl ] && mv data/spend.jsonl data/spend.published.jsonl || true; }
echo "+ mv data/spend.jsonl data/spend.published.jsonl  (if present)"
# 1. Pilot on 5 development writers (the prompt stays Episode 1's template v3 whatever it shows)
run $PY run_inkling.py dev --authors 5 --tag pilot
# 2. Held-out: 30 writers x 4 briefs x 6 signals, reasoning off; then the exploratory reasoning-low arm
run $PY run_inkling.py heldout
run $PY run_inkling.py heldout --arms k0,k8_rand,note --reasoning low
# 3. Episode 1's rulers: voice position (strict, loose, 80 words), specificity, detector, copying, loops
run $PY score_inkling.py data/drafts/inkling-2-mini__heldout.jsonl data/drafts/inkling-2-mini-reason-low__heldout.jsonl
run $PY score_inkling.py --ep1-cut80
run $PY score_inkling.py --addressed
# The scorer check: Episode 1's Gemini drafts through this scorer must give its published row
run $PY score_inkling.py ../sounds-like-me/data/drafts/gemini-4-carbon__heldout.jsonl
# 4. Honesty extractor + rule layer (Qwen3-32B on Modal)
run $PY score_inkling.py --honesty data/drafts/inkling-2-mini__heldout.jsonl data/drafts/inkling-2-mini-reason-low__heldout.jsonl
# 5. Leaderboard, paired contrasts, predictions, spend; figures; page data
run $PY analyze.py
run $PY make_figures.py
run $PY export_pages.py
