#!/usr/bin/env bash
# Rerun every number in the adapter self-interpretation replication note. COSTS MONEY (one Modal H100, about $15);
# run.py refuses any launch past the $25 cap in results/spend.json.
#
#   experiments/repl-adapter-self-interp/reproduce.sh            # everything
#   experiments/repl-adapter-self-interp/reproduce.sh --dry-run  # print the commands
#   experiments/repl-adapter-self-interp/reproduce.sh --scores   # only the summary and figures, from results/
#
# Needs: Modal (profile credits30, secret huggingface-secret with access to meta-llama/Llama-3.1-8B-Instruct).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
D=experiments/repl-adapter-self-interp
export MODAL_PROFILE=credits30 MODAL_IMAGE_BUILDER_VERSION=2025.06

steps=(
  ".venv/bin/modal deploy $D/modal_app.py"
  "$PY $D/run.py prep"          # decoder vectors of 45,418 labelled latents; SAE reconstruction check
  "$PY $D/run.py train"         # scalar-affine adapter (released recipe), then the shuffled-label control
  "$PY $D/run.py valloss"       # Table-1-style val losses
  "$PY $D/run.py eval smoke"    # 20 latents: pipeline check (not reported)
  "$PY $D/run.py eval main"     # the 500 pre-registered latents
)
scores=(
  "$PY $D/analyse.py main"
  "$PY $D/make_figures.py"
)
if [[ "${1:-}" == "--dry-run" ]]; then printf '%s\n' "${steps[@]}" "${scores[@]}"; exit 0; fi
if [[ "${1:-}" != "--scores" ]]; then for s in "${steps[@]}"; do echo "+ $s"; eval "$s"; done; fi
for s in "${scores[@]}"; do echo "+ $s"; eval "$s"; done
