#!/bin/zsh
# Reproduce every number in the note "Does judge-RL erode plurality?" (Episode 3 of the pluralism series).
# COSTS MONEY: about $90 of Modal GPU time (H100 training, L40S judges and scoring) and under $1 of OpenRouter.
# Run from the repository root. --dry-run prints the commands without running them.
set -e
DRY=0; [[ "$1" == "--dry-run" ]] && DRY=1
PY=.venv/bin/python; MODAL=.venv/bin/modal; X=experiments/pluralism-judge-rl
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }
RUNS="single-creative-s0,single-creative-s1,jury-creative-s0,jury-creative-s1,shuffled-creative-s0,bon-creative-s0,bon-creative-s1,single-advice-s0,single-advice-s1,jury-advice-s0,jury-advice-s1,shuffled-advice-s0,bon-advice-s0,bon-advice-s1,clean-s0,clean-s1"

# 1. fixed prompt sets (committed in data/; rebuilding is deterministic)
run $PY $X/data_prep.py
run $MODAL run $X/modal_rl.py::neutral_base            # skipped if data/neutral_base.jsonl exists
# 2. the open judge, served on L40S, and its validation (autoresearch ledger: results/judge_search/results.tsv)
run $MODAL deploy $X/modal_judge.py
run $MODAL run $X/modal_score.py::sample_base
run $PY $X/judge_search.py --candidate microsoft/phi-4 --note "reproduce"
run $PY $X/judge_search.py --candidate microsoft/phi-4 --confirm --note "reproduce (held-out)"
# 3. pilot (exploration only) and the 16 confirmatory runs, spawned on the deployed trainer so a client failure
#    cannot cancel them
run $MODAL run $X/modal_rl.py::train_many --config configs/pilot.json
run $MODAL deploy $X/modal_rl.py
run $PY $X/modal_rl.py configs/full.json $RUNS   # config path is relative to this folder
# 4. score every checkpoint, and the base model twice (the do-nothing floor)
run $MODAL run $X/modal_score.py::score --base-seeds 0,1 --split 2
run $MODAL run $X/modal_score.py::score --runs $RUNS --split 4
run $MODAL run $X/modal_score.py::survey_ep1 --base-seeds 0,1 --runs $RUNS --split 4   # Episode 1's survey eval
run $PY $X/shared_measures.py                                                          # Episode 2's measures.py
# 5. analysis, judge-based measures, figures
run $PY $X/analyse.py --config configs/full.json
run $PY $X/extras.py --config configs/full.json
run $PY $X/make_figures.py
run $PY $X/make_cover.py
run $MODAL app stop -y pluralism-judge-rl-judge
