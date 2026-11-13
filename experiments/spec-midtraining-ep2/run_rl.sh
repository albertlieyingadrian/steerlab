#!/usr/bin/env bash
# The RL test's runs in parallel waves, holding .claude/mac-busy-msm2.lock (main checkout) while the laptop renders.
# COSTS MONEY (Modal L40S, about $2 per 100-step run). Usage, from anywhere:
#     experiments/spec-midtraining-ep2/run_rl.sh H_s0 T_s0 H_s1 T_s1      # one wave, one run per start model
#     experiments/spec-midtraining-ep2/run_rl.sh B_s0@4                    # the untrained model (one copy, B_s0), RL seed 4, run B_s4
# Each run's seed is the start model's seed (H_s3 -> --seed 3); logs go to experiments/spec-midtraining-ep2/logs/rl_<run>.log.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${HERE%%/.claude/worktrees/*}"
ROOT="$(cd "$ROOT" && git rev-parse --show-toplevel)"
mkdir -p "$HERE/logs" "$ROOT/.claude"
LOCK="$ROOT/.claude/mac-busy-msm2.lock"
# one line per process (waves and judge_rl.py may overlap); each removes only its own line, the last removes the file
echo "pid $$: spec-midtraining-ep2 RL test ($*) rendering on this Mac since $(date)" >> "$LOCK"
release() {
  if [ -f "$LOCK" ]; then
    grep -vF "pid $$:" "$LOCK" > "$LOCK.$$.tmp" || true
    if [ -s "$LOCK.$$.tmp" ]; then mv "$LOCK.$$.tmp" "$LOCK"; else rm -f "$LOCK" "$LOCK.$$.tmp"; fi
  fi
}
trap release EXIT
pids=()
for arg in "$@"; do
  start="${arg%@*}"
  seed="$(echo "$start" | sed -E 's/^[A-Z]+_s([0-9]+).*/\1/')"
  name="$start"
  if [[ "$arg" == *@* ]]; then
    [[ "$start" == B_* ]] || { echo "@seed is only for the untrained model (B_s0@k)"; exit 1; }
    seed="${arg#*@}"
    name="${start%_s*}_s$seed"
  fi
  (cd "$HERE" && MODAL_PROFILE=albertlieyingadrian "$ROOT/.venv/bin/modal" run rl_modal.py \
      --start "$start" --run "${RUN_PREFIX:-}$name" --seed "$seed" ${RL_ARGS:-} > "$HERE/logs/rl_${RUN_PREFIX:-}$name.log" 2>&1) &
  pids+=($!)
  sleep 20  # stagger the Chromium pools and CLIP loads
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
