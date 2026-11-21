#!/bin/zsh
# Reproduce every number in the note "Can a brand book become a checkable spec?" (The Taste Stack, Episode 5).
# COSTS MONEY: a few dollars of Modal H100 time (vLLM generation). Everything else runs on a laptop (Playwright's
# Chromium renders and reads the pages). Run from the repository root. --dry-run prints the commands.
set -e
DRY=0; [[ "$1" == "--dry-run" ]] && DRY=1
PY=.venv/bin/python; MODAL=.venv/bin/modal; X=experiments/brand-spec
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

# 0. (optional) rebuild the vendored brand books from pinned upstream clones:
#    alphagov/govuk-design-system 88066fd8, uswds/uswds-site af13d80f, uswds/uswds 029d6d79
# run $PY $X/vendor/make_sources.py GOVUK_CLONE USWDS_SITE_CLONE USWDS_CLONE
# 1. formalise: both specs from the vendored lines, with the spec lint (deterministic)
run $PY $X/formalise.py
# 2. pages: generate, render, extract and check every page set; freeze them (hashes in data/manifest.json)
run $PY $X/dataset.py
# 3. the model test sets (page audits, identification, application), frozen before any reply
run $PY $X/questions.py
# 4. every ruler against its pass bar (exits 1 if one is missed); --real also loads the two brands' home pages
run $PY $X/ruler.py --real
# 5. the copy check (deterministic; the gate and its threshold are chosen on dev)
run $PY $X/copycheck.py
# 6. replies: dev tuning (Qwen3-8B), then held-out for both models, and a second sample of the page audits
B="t1:govuk:dev:a,t1:govuk:dev:b,t2:govuk:dev:b,t3:govuk:dev:b,t3c:govuk:dev:b"
B="$B,t1:uswds:dev:a,t1:uswds:dev:b,t2:uswds:dev:b,t3:uswds:dev:b,t3c:uswds:dev:b"
run $MODAL run $X/modal_gen.py --model qwen3-8b --jobs $B
H="${B//:dev:/:heldout:}"
run $MODAL run $X/modal_gen.py --model qwen3-8b --jobs $H
run $MODAL run $X/modal_gen.py --model qwen3-8b --seed 1 --jobs t1:govuk:heldout:b,t1:uswds:heldout:b
run $MODAL run $X/modal_gen.py --model qwen3-32b --jobs $H
# 7. score, figures, the interactive's data, the cover
run $PY $X/score.py --split dev
run $PY $X/score.py
run $PY $X/make_figures.py
run $PY $X/make_demo.py
run $PY $X/make_cover.py
