#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/natural-sandbagging/reproduce.sh --dry-run    print the commands only
#   experiments/natural-sandbagging/reproduce.sh              the laptop steps (about 5 hours on an M5 Max, 128 GB)
#   experiments/natural-sandbagging/reproduce.sh --modal      also the Qwen3-32B run on one Modal H100 (about $2)
#
# WMDP and MMLU (both MIT) download from Hugging Face on first use; Qwen3-8B and Qwen3-32B are Apache 2.0. Reasoning
# traces and activations go to ~/.cache/steerlab/natural-sandbagging/ (NATURAL_SANDBAGGING_WORK moves it), never into
# the repository: results/ holds item IDs and scores. Generation is append-only, so a stopped run resumes. Some scripts
# print ❌ lines and exit 1; those are hypotheses the note reports as not supported, so the run carries on past them.
set -uo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
MODAL_RUN=0
for a in "$@"; do
  [ "$a" = "--dry-run" ] && DRY_RUN=1
  [ "$a" = "--modal" ] && MODAL_RUN=1
done
PY=${PYTHON:-.venv/bin/python}
MODAL=${MODAL:-.venv/bin/modal}
E=experiments/natural-sandbagging
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then
    "$@" || echo "  (exit $?: see the ❌ lines above)"
  fi
}

echo "# 2-3: the item split (seed 1919) and the sizing pilot (no framing)"
run "$PY" "$E/items.py"
run "$PY" "$E/pilot.py"

echo "# 3: Qwen3-8B, 7,200 core generations, then 1,200 under the motive framing"
run "$PY" "$E/generate.py" --run core
run "$PY" "$E/generate.py" --run motive

echo "# 4.4: the eval-awareness direction on held-out items, then nine steering arms"
run "$PY" "$E/direction.py"
run "$PY" "$E/steer.py"

if [ "$MODAL_RUN" = 1 ] || [ "$DRY_RUN" = 1 ]; then
  echo "# 4.2 (only with --modal; costs money): Qwen3-32B, weights on a CPU-only container, then one capped H100 call"
  run "$MODAL" run "$E/modal_32b.py::download"
  run "$MODAL" run "$E/modal_32b.py::main" --tag pilot
fi

echo '# 4.3: the judge on every seed-0 trace (Qwen3-8B, thinking off). The note used the Modal judge (plan amendment 2,'
echo '#      about $1, after an agreement check on 50 traces); without --modal the laptop judge runs instead (about 90 min)'
if [ "$MODAL_RUN" = 1 ] || [ "$DRY_RUN" = 1 ]; then
  run "$PY" "$E/judge_agreement.py" --local
  run "$MODAL" run "$E/modal_judge.py::download"
  run "$MODAL" run "$E/modal_judge.py::main" --check work --sources stage2
  run "$PY" "$E/judge_agreement.py" --compare
  run "$MODAL" run "$E/modal_judge.py::main" --sources core,motive,steer
fi
if [ "$MODAL_RUN" = 0 ] || [ "$DRY_RUN" = 1 ]; then
  for s in core motive steer stage2; do
    run "$PY" "$E/judge.py" --source "$s"
  done
fi

echo "# 4: scoring, the reasoning reads, the steering test, the scoreboard and the figures"
run "$PY" "$E/score.py"
for s in core motive stage2; do
  run "$PY" "$E/reasoning.py" --source "$s"
done
run "$PY" "$E/steer_summary.py"
run "$PY" "$E/score_32b.py"
run "$PY" "$E/scoreboard.py"
run "$PY" "$E/make_figures.py"
run "$PY" "$E/make_art.py"
