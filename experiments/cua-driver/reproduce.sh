#!/bin/bash
# Every command behind the note's numbers, in order.   ./reproduce.sh [--dry-run]
# Needs: an Apple-silicon Mac with Lume and Docker (colima); a clone of Scale's AgentEnv port
# (github.com/earakely-scale/agentenv-cua-driver-bench) at 3f8210e with AgentEnv 0.9.1506 in its .venv, at $PORT;
# a LiteLLM proxy on 127.0.0.1:4000 in front of the Anthropic API (its key in $LITELLM_API_KEY).
# The analysis step alone (the last two lines) needs nothing but Python with scipy and matplotlib.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-$HOME/oss/agentenv-cua-driver-bench}
run() { echo "+ $*"; [ "${1:-}" = "" ] || [ "$DRY" = 1 ] || eval "$@"; }
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
# 1. Build the frozen driver (0.5.1) and install it as an arm in the port (driver env, computers, tasks, schedules).
run "python3 '$HERE/agentenv/sync.py' --port '$PORT'"
# 2. The interleaved A/B schedule: run 1 of all 18 variants for both arms, then run 2, ...
run "python3 - <<'PY'
import re
from collections import OrderedDict
port = '$PORT'
A = [l.strip() for l in open(f'{port}/evals/macos-ourdriver-bench-v2-trials.txt') if l.strip()]
B = [l.strip() for l in open(f'{port}/evals/macos-cua-driver-bench-v2-trials.txt') if l.strip()]
ga, gb = OrderedDict(), OrderedDict()
for a, b in zip(A, B):
    ga.setdefault(re.sub(r'-r\\d-', '-', a), []).append(a)
    gb.setdefault(re.sub(r'-r\\d-', '-', b), []).append(b)
out = [t for run in range(5) for va, vb in zip(ga, gb) for t in (ga[va][run], gb[vb][run])]
open(f'{port}/evals/final-ab-interleaved.txt', 'w').write('\\n'.join(out) + '\\n')
PY"
# 3. Two runners share the schedule (set max_running = 2 in the port's .agentenv/config.toml first).
run "PORT='$PORT' '$HERE/tools/run_par.sh' final-ab-opus '$PORT/evals/final-ab-interleaved.txt' -- --model anthropic/claude-opus-5-5 &"
run "PORT='$PORT' '$HERE/tools/run_par.sh' final-ab-opus '$PORT/evals/final-ab-interleaved.txt' -- --model anthropic/claude-opus-5-5; wait"
# 4. Archive every trial (grade result, transcript, snapshots) and write results/trials.csv.
run "PORT='$PORT' python3 '$HERE/tools/collect_trials.py' final-ab-opus && cp \"\${CUA_RUNS:-\$HOME/oss/cua-runs}/final-ab-opus/trials.csv\" '$HERE/results/'"
# 5. Every number and figure in the note, from results/trials.csv.
run "python3 '$HERE/analyse.py'"
run "python3 '$HERE/make_cover.py'"
