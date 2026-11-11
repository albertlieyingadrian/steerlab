#!/usr/bin/env bash
# Rerun the note. The run steps need the ATCO2 test set 1h in data/raw/xml/ (see README.md), an Anthropic key, a
# Tinker key, an OpenAI key (GPT baselines only), a Kaggle token (KAGGLE_API_TOKEN) and a clone of the GMAPS repository
# for its published predictions (GMAPS=<path to its results/test>). Everything cost about $62 at list prices plus about
# 13.5 free Kaggle GPU-hours (2026-11-11). With --scores, only the steps that read saved results run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
GMAPS=${GMAPS:-../../../GMAPS_Fireworks/results/test}
STEPS=(
  "$PY atc.py"
  "$PY run.py --model qwen --split test --shots 5 --name qwen-fs5"
  "$PY run.py --model claude-opus-5-5:medium --split test --shots 5 --max-tokens 16000"
  "$PY run.py --model claude-opus-5-5:medium --split test --shots 20 --max-tokens 16000"
  "$PY run.py --model claude-fable-5-2:medium --split test --shots 5 --max-tokens 16000"
  "$PY run.py --model claude-fable-5-2:medium --split test --shots 20 --max-tokens 16000"
  "$PY run.py --model gpt-5.4:medium --split test --shots 20 --max-tokens 6000"
  "$PY harness.py role --run test__gpt-5.4-medium-fs20"
  "$PY harness.py role --run test__qwen-fs5"
  "$PY external.py fetch"
  "$PY external.py label --model q397 --n 3000 --shots 20 --sources atcosim --tag q397"
  "$PY external.py synth --n 2500"
  "$PY external.py augment --n 2000"
  "$PY external.py readbacks --n 2000"
  "$PY train.py mix --name pfa --pseudo atcosim.q397 --augment"
  "$PY train.py mix --name pfr --pseudo atcosim.q397 --readbacks"
  "$PY kaggle_job.py dataset"
  "$PY kaggle_job.py train --mix pfa --name q14-pfa"
  "$PY kaggle_job.py fetch --slug gmaps-q14-pfa"
  "$PY kaggle_job.py infer --run q14-pfa --ckpt epoch1 --split test"
  "$PY kaggle_job.py train --mix pfr --name q14-pfr"
  "$PY kaggle_job.py fetch --slug gmaps-q14-pfr"
  "$PY kaggle_job.py rft --run q14-pfr --ckpt epoch1 --name q14-pfr-rft --lr 5e-5 --k 8"
  "$PY kaggle_job.py fetch --slug gmaps-q14-pfr-rft"
  "$PY kaggle_job.py infer --run q14-pfr --ckpt epoch1 --split test"
  "$PY kaggle_job.py fetch --slug gmaps-q14-pfr-epoch1-test"
  "$PY summarize.py --gmaps $GMAPS"
)
SCORES=("$PY make_figures.py" "$PY make_cover.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=()
fi
STEPS+=("${SCORES[@]}")
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  eval "$step"
done
