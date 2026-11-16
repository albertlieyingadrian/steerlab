#!/bin/bash
# Like run_seq.sh, but several copies can share one schedule: each trial is claimed (mkdir claims/<n>) by the
# first copy to reach it, so N copies on N VMs work through the list in order without running a trial twice.
#   run_par.sh <label> <task list file> [-- extra agent-env args]
#
# Guards (added after final 4's first resume, when leftover VMs held both slots and the runner marked 166 instant
# harness failures as done):
# - before each launch, wait until fewer than MAX_VMS macOS VMs are running (lume ls);
# - a trial whose last attempt is a harness failure is NOT marked done: its claim is released, it is logged in
#   harness.tsv, and the trial stays in the schedule to be rerun in order;
# - after 3 harness failures in a row (across trials, any copy), every copy stops and says so in run.log;
# - an attempt with no result line at all counts as a harness failure (added after 11 image-build failures at
#   final 4's second resume were marked done).
set -u
LABEL=$1; shift
PORT=${PORT:-$HOME/oss/agentenv-cua-driver-bench}   # a clone of Scale's port with this arm installed (agentenv/sync.py)
LOCK=${BUSY_LOCK:-/nonexistent}   # optional: while this file exists, no new trial starts
MAX_VMS=${MAX_VMS:-2}
RUNS=${CUA_RUNS:-$HOME/oss/cua-runs}; OUT=$RUNS/$LABEL; mkdir -p "$OUT/claims"
TASKS=(); EXTRA=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
  if [ -f "$1" ]; then while read -r t; do [ -n "$t" ] && TASKS+=("$t"); done < "$1"; else TASKS+=("$1"); fi
  shift
done
export LITELLM_API_KEY=${LITELLM_API_KEY:-$(cat "$RUNS/proxy/master.key")}   # the local LiteLLM proxy's key
HARNESS_RE="failed at step '(env|arm_tools|sandbox|box|pack|agent|skills|record|arm|snapshot|trial)'|DeployRefused|API Error: (5[0-9][0-9]|overloaded)|overloaded_error|Internal server error|can't be read"
running_vms() { lume ls 2>/dev/null | grep -c " running " ; }
cd "$PORT"
i=0
for t in "${TASKS[@]}"; do
  i=$((i + 1)); n=$(printf "%03d" $i)
  [ -e "$OUT/STOP" ] && { echo "$(date) $$ stopping: STOP present" >> "$OUT/run.log"; exit 4; }
  if grep -q "^$n	" "$OUT/done.txt" 2>/dev/null; then continue; fi
  mkdir "$OUT/claims/$n" 2>/dev/null || continue          # another copy has it
  while [ -e "$LOCK" ]; do echo "$(date) lock present, waiting" >> "$OUT/run.log"; sleep 600; done
  while [ "$(running_vms)" -ge "$MAX_VMS" ]; do sleep 20; done
  sleep $((RANDOM % 10))                                  # the two copies don't check capacity in the same instant
  while [ "$(running_vms)" -ge "$MAX_VMS" ]; do sleep 20; done
  start=$(date +%s)
  harness=0
  for attempt in 1 2 3; do
    log="$OUT/$n-$t.log"; [ $attempt -gt 1 ] && log="$OUT/$n-$t.retry$attempt.log"
    .venv/bin/agent-env run . --task "$t" ${EXTRA[@]+"${EXTRA[@]}"} > "$log" 2>&1
    code=$?
    line=$(grep -E "^  tasks/$t.json" "$log" | tail -1 | sed 's/^ *//')
    # No result line at all (e.g. the image build failed before the trial: "docker buildx isn't installed") is a
    # harness failure too; final 4 scored 11 of these as fails before this check existed.
    if [ -z "$line" ] || echo "$line" | grep -qE "$HARNESS_RE"; then
      [ -z "$line" ] && line="no result line: $(grep -m1 -E '^Error' "$log" | cut -c1-200)"
      harness=1
      printf '%s\t%s\t%s\tattempt %s harness failure: %s\n' "$(date +%FT%T)" "$n" "$t" "$attempt" "${line:0:300}" >> "$OUT/retries.tsv"
      while [ "$(running_vms)" -ge "$MAX_VMS" ]; do sleep 20; done
      continue
    fi
    harness=0
    break
  done
  if grep -qE "authentication_error|invalid x-api-key|API Error: 401|usage limit|credit balance is too low" "$log"; then
    echo "$(date) stopped at $n $t: the API key no longer works" >> "$OUT/run.log"; rmdir "$OUT/claims/$n"; exit 3
  fi
  if [ $harness = 1 ]; then
    # No result: release the trial for a later rerun, and stop everything after 3 in a row.
    printf '%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$n" "$t" "${line:0:300}" >> "$OUT/harness.tsv"
    rmdir "$OUT/claims/$n"
    echo 1 >> "$OUT/harness_streak"
    if [ "$(wc -l < "$OUT/harness_streak")" -ge 3 ]; then
      touch "$OUT/STOP"
      echo "$(date) $$ stopping: 3 harness failures in a row (last $n $t)" >> "$OUT/run.log"; exit 5
    fi
    continue
  fi
  : > "$OUT/harness_streak"
  printf '%s\t%s\t%s\t%ss\t%s\n' "$(date +%FT%T)" "$t" "$code" "$(( $(date +%s) - start ))" "$line" >> "$OUT/summary.tsv"
  printf '%s\t%s\n' "$n" "$t" >> "$OUT/done.txt"
done
echo "done $(date) ($$)" >> "$OUT/run.log"
