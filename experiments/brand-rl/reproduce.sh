#!/bin/zsh
# Reproduce every number in the note "An RL environment for 'on brand, not slop, not a copy'" (The Taste Stack, Episode 6).
# COSTS MONEY: about $300 of Modal H100 time (seven 200-step GRPO runs of about ten hours each, plus sampling).
# The rulers, analysis and figures run on a laptop (Playwright's Chromium). Run from the repository root.
# --dry-run prints the commands. Episode 5 (experiments/brand-spec) and Episode 3's caches must be in place.
set -e
DRY=0; [[ "$1" == "--dry-run" ]] && DRY=1
PY=.venv/bin/python; MODAL=.venv/bin/modal; X=experiments/brand-rl
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }
# Modal jobs write to the taste-ep6 volume; the local steps read the same paths under ep6.CACHE
C=$HOME/.cache/steerlab/brand-rl
pull() { for f in "$@"; do run mkdir -p "$C/${f:h}"; run $MODAL volume get taste-ep6 "$f" "$C/$f" --force; done; }

# 1. freeze Episode 3's markup and screenshot probes (asserts the refit matches its held-out AUROC)
run $PY $X/slop_fit.py
# 2. the Modal app (training, sampling, steering, screenshot probe)
run $MODAL deploy $X/modal_rl.py
# 3. base pages: the dev exploration (family split and novelty scale, already fixed in configs/) and the monitor briefs
run $MODAL run $X/modal_rl.py::jobs --spec $X/configs/explore_base.json
pull samples/base_dev_explore.pkl.gz samples/base_monitor.pkl.gz
# 4. every ruler against its pass bar (exits 1 if one is missed)
run $PY $X/ruler.py
# 5. the monitor: constructed hacks of the base model's honest pages, teacher-forced, then the direction and probe
run $PY $X/hacks.py build
run $MODAL run $X/modal_rl.py::jobs --spec $X/configs/monitor_tf.json
pull samples/monitor_tf.pkl.gz
run $PY $X/hacks.py fit
# 6. the use test (steering the base model along the hack direction vs norm-matched random directions)
run $PY $X/steer_jobs.py
run $MODAL run $X/modal_rl.py::steer --spec $X/configs/steer.json
pull steer/steer_select_monitor.pkl.gz steer/steer_dev_a0.1.pkl.gz steer/steer_dev_a0.25.pkl.gz \
    steer/steer_dev_a0.5.pkl.gz
# 7. the seven confirmatory runs (detached), then the driver: respawns, dev-best, held-out sampling, screenshot probe
run $PY $X/launch.py configs/main.json
run $PY $X/drive.py configs/main.json
# 8. numbers and figures
run $PY $X/analyze.py configs/main.json
run $PY $X/make_figures.py configs/main.json
