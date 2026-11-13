#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/auditbench/reproduce.sh --dry-run    print the commands only
#   experiments/auditbench/reproduce.sh              run them (about 3 hours on an M-series laptop with 64 GB or more)
#
# Downloads about 72 GB into the Hugging Face cache: Qwen/Qwen3-14B and 42 of AuditBench's LoRA adapters
# (huggingface.co/auditing-agents). AuditBench's prompts are read from github.com/safety-research/auditing-agents at a
# pinned commit into the work folder ($AUDITBENCH_WORK, default ~/.cache/steerlab/auditbench); only their hashes are
# kept here. The judge needs OPENAI_API_KEY (well under $1 of gpt-4.1-mini; calls are cached).
set -uo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/auditbench

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then
    "$@" || { echo "  (exit $?)"; exit 1; }
  fi
}

echo "# 2: AuditBench's prompts at a pinned commit; the base model and 42 adapters"
run "$PY" "$E/fetch.py"

echo "# 3.1: the base model: behaviour directions, the two probes and the quiz; then each of the 42 adapters"
run "$PY" "$E/run_base.py" --stage=read
run "$PY" "$E/run_organisms.py" --stage=read

echo "# 3.2: replies: the base model under each quirk prompt (the judge check), then each adapter at alpha 0 and 1"
run "$PY" "$E/run_base.py" --stage=replies
run "$PY" "$E/run_organisms.py" --stage=replies

echo "# 3.3: noise calibration and the prompted sandbagger under noise (the organisms' noise pass was not run:"
echo "#      every model scored 60/60, so none could improve; run_organisms.py --stage=noise adds it)"
run "$PY" "$E/run_base.py" --stage=noise

echo "# 3.4: the judge ranks the 14 behaviours for every set of replies"
run "$PY" "$E/judge.py"

echo "# 4: every detector against the answer key, with bootstrap intervals and the pre-registered criteria"
run "$PY" "$E/score.py"

echo "# figures"
run "$PY" "$E/make_figures.py"

echo "# Llama-3.3-70B half (PREREGISTRATION-70B.md): two H100s on Modal, about \$20; needs Hugging Face access to"
echo "#   meta-llama/Llama-3.3-70B-Instruct. Runs on a deployed app, so it survives the laptop going offline."
run "$PY" "$E/fetch.py" --prompts-only
run "${MODAL:-.venv/bin/modal}" deploy "$E/modal_70b.py"
run "$PY" "$E/modal_70b.py" check
run "$PY" "$E/modal_70b.py" download
run "$PY" "$E/modal_70b.py" submit smoke
run "$PY" "$E/modal_70b.py" wait smoke
run "$PY" "$E/modal_70b.py" submit full
run "$PY" "$E/modal_70b.py" wait full
run "$PY" "$E/modal_70b.py" pull full
run "$PY" "$E/score_70b.py" --tag=full
