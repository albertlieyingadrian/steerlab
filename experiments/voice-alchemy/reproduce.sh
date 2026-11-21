#!/usr/bin/env bash
# Every step behind the Voice Alchemy note, in order. GPU steps run on Modal H100s (about $10 in all); judge steps
# call OpenRouter (about $4). `--dry-run` prints the commands; `--scores` reruns only the local analysis.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
EXP=experiments/voice-alchemy
MODAL="env MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian} .venv/bin/modal run"
CT="CIRCUIT_TRACER_SRC=${CIRCUIT_TRACER_SRC:-../oss/circuit-tracer}"

steps=(
  "cd $EXP && ../../$PY judge.py ruler"
  "$MODAL $EXP/modal_app.py --stage find"
  "cd $EXP && ../../$PY pipeline.py jobs dev1"
  "$MODAL $EXP/modal_app.py --stage gen --jobs data/jobs_dev1.jsonl --out results/gen_dev1.jsonl"
  "cd $EXP && ../../$PY pipeline.py score dev1"
  "cd $EXP && ../../$PY pipeline.py jobs dev2"
  "$MODAL $EXP/modal_app.py --stage gen --jobs data/jobs_dev2.jsonl --out results/gen_dev2.jsonl"
  "cd $EXP && ../../$PY pipeline.py score dev2"
  "cd $EXP && ../../$PY pipeline.py jobs dev3"
  "$MODAL $EXP/modal_app.py --stage gen --jobs data/jobs_dev3.jsonl --out results/gen_dev3.jsonl"
  "cd $EXP && ../../$PY pipeline.py score dev3"
  "env $CT $MODAL $EXP/circuits.py"
  "cd $EXP && ../../$PY pipeline.py jobs dev4"
  "$MODAL $EXP/modal_app.py --stage gen --jobs data/jobs_dev4.jsonl --out results/gen_dev4.jsonl"
  "cd $EXP && ../../$PY pipeline.py score dev4"
  "cd $EXP && ../../$PY analyze.py dev4"
  "cd $EXP && ../../$PY pipeline.py select x"
  "cd $EXP && ../../$PY pipeline.py jobs pilot"
  "$MODAL $EXP/modal_app.py --stage gen --jobs data/jobs_pilot.jsonl --out results/gen_pilot.jsonl"
  "cd $EXP && ../../$PY pipeline.py score pilot"
  "cd $EXP && ../../$PY analyze.py pilot"
  "cd $EXP && ../../$PY pipeline.py jobs heldout"
  "$MODAL $EXP/modal_app.py --stage gen --jobs data/jobs_heldout.jsonl --out results/gen_heldout.jsonl"
  "cd $EXP && ../../$PY pipeline.py score heldout"
  "cd $EXP && ../../$PY recheck.py"
  "env $MODAL $EXP/open_judge_qwen32b_check.py"
  "env MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian} .venv/bin/modal deploy $EXP/open_judge.py"
  "cd $EXP && ../../$PY judge.py ruler llama gemma27 mistral phi4"
  "env VA_MODEL=qwen $MODAL $EXP/modal_app.py --stage find"
)
scores=(
  "cd $EXP && ../../$PY analyze.py heldout"
  "cd $EXP && ../../$PY recheck.py --analyse-only"
  "cd $EXP && ../../$PY build_table.py"
  "cd $EXP && ../../$PY make_figures.py"
  "cd $EXP && ../../$PY make_cover.py"
  "cd $EXP && ../../$PY record_demo.py"
)

run() { echo "+ $1"; [[ "${DRY:-0}" == 1 ]] || (eval "$1"); }
case "${1:-}" in
  --dry-run) DRY=1; for s in "${steps[@]}" "${scores[@]}"; do run "$s"; done ;;
  --scores) for s in "${scores[@]}"; do run "$s"; done ;;
  *) for s in "${steps[@]}" "${scores[@]}"; do run "$s"; done ;;
esac
