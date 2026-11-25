#!/usr/bin/env bash
# Rerun every number in the where-does-the-evil-go note. COSTS MONEY (Modal H100 time and a few dollars of
# gpt-4.1-mini through OpenRouter); the driver refuses any launch past the $150 cap in results/spend.json.
#
#   experiments/where-does-the-evil-go/reproduce.sh            # everything
#   experiments/where-does-the-evil-go/reproduce.sh --dry-run  # print the commands
#   experiments/where-does-the-evil-go/reproduce.sh --scores   # only the summary and figures, from cached replies
#
# Needs: Modal (profile albertlieyingadrian, secret huggingface-secret), OPENROUTER_API_KEY_2 in .env.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RUN="$PY experiments/where-does-the-evil-go/run.py"
export MODAL_PROFILE=albertlieyingadrian

steps=(
  ".venv/bin/modal deploy experiments/where-does-the-evil-go/modal_app.py"
  ".venv/bin/modal deploy experiments/where-does-the-evil-go/modal_judge.py"
  "$RUN forkdata"        # the post's fork at commit 33ba92c: its judged replies and the paper's datasets
  "$RUN judgecheck"      # our judge vs the post's gpt-4.1-mini scores on 1,200 of its replies
  "$RUN base"            # base sweep: 4 traits x coefs 0-2.5, 3 random directions x coefs 1-2.5
  "$RUN finetune"        # six fine-tunes
  "$RUN collect"         # wait for them
  "$RUN ftsweep"         # re-extraction + sweep along the old vector, per fine-tune
  "$RUN extractbase"     # the base model's re-extraction, seeds 0 and 1
  "$RUN collect"
  "$RUN judge"           # score every reply
  "$RUN vectors"         # re-extracted vectors, cosines, evil on request, projections
  "$RUN steernew"        # steer each model with its own re-extracted vector
  "$RUN collect"
  "$RUN judge"
  "$RUN gain"            # KL-gain test
  "$RUN crosscheck"      # gpt-4.1-mini on 300 of our replies
  "$RUN headline_gpt"    # the headline cells in full with gpt-4.1-mini
  "$RUN summarise"
  "$PY experiments/where-does-the-evil-go/make_figures.py"
)
scores=("$RUN vectors" "$RUN summarise" "$PY experiments/where-does-the-evil-go/make_figures.py")

if [[ "${1:-}" == "--dry-run" ]]; then printf '%s\n' "${steps[@]}"; exit 0; fi
if [[ "${1:-}" == "--scores" ]]; then steps=("${scores[@]}"); fi
for s in "${steps[@]}"; do echo "+ $s"; eval "$s"; done
.venv/bin/modal app stop -y where-does-the-evil-go-judge || true
