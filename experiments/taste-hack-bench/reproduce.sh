#!/bin/zsh
# Reproduce every number in The Taste Stack Episode 8 note (the taste-reward hack benchmark).
# Free: CPU only, about a minute, no model or API calls. Run from the repository root.
# --dry-run prints the commands without running them.
set -e
DRY=0; [[ "$1" == "--dry-run" ]] && DRY=1
PY=.venv/bin/python; X=experiments/taste-hack-bench
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

# 1. the four cheap detectors pass their known-answer checks
run $PY $X/detectors.py
# 2. entries.jsonl and artefacts/ from entries.py and the source notes' shipped outputs (fails if a quote is not in its note)
run $PY $X/build.py
# 3. the detection table and the scored predictions
run $PY $X/baselines.py
# 4. the gallery's data, Table 2 and the cover
run $PY $X/make_report.py
