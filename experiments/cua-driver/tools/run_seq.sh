#!/bin/bash
# Run AgentEnv tasks one at a time (one macOS VM), honouring the FinanceMath lock between tasks.
#   run_seq.sh <label> <task list file or task names...> [-- extra agent-env args]
# Logs to $CUA_RUNS/<label>/<n>-<task>.log and a summary line per task in summary.tsv.
set -u
LABEL=$1; shift
PORT=${PORT:-$HOME/oss/agentenv-cua-driver-bench}   # a clone of Scale's port with this arm installed (agentenv/sync.py)
LOCK=${BUSY_LOCK:-/nonexistent}   # optional: while this file exists, no new trial starts
RUNS=${CUA_RUNS:-$HOME/oss/cua-runs}; OUT=$RUNS/$LABEL; mkdir -p "$OUT"
TASKS=(); EXTRA=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
  if [ -f "$1" ]; then while read -r t; do [ -n "$t" ] && TASKS+=("$t"); done < "$1"; else TASKS+=("$1"); fi
  shift
done
export LITELLM_API_KEY=${LITELLM_API_KEY:-$(cat "$RUNS/proxy/master.key")}   # the local LiteLLM proxy's key
cd "$PORT"
i=0
for t in "${TASKS[@]}"; do
  i=$((i + 1)); n=$(printf "%03d" $i)
  if grep -q "^$n	" "$OUT/done.txt" 2>/dev/null; then continue; fi   # resume: skip trials already run
  while [ -e "$LOCK" ]; do echo "$(date) lock present, waiting" >> "$OUT/run.log"; sleep 600; done
  start=$(date +%s)
  # A harness failure (the computer never deployed, a setup step failed, an API 5xx/overload) is retried with the
  # same trial id, up to 2 more times, and logged; a task failure is never retried.
  for attempt in 1 2 3; do
    log="$OUT/$n-$t.log"; [ $attempt -gt 1 ] && log="$OUT/$n-$t.retry$attempt.log"
    .venv/bin/agent-env run . --task "$t" ${EXTRA[@]+"${EXTRA[@]}"} > "$log" 2>&1
    code=$?
    line=$(grep -E "^  tasks/$t.json" "$log" | tail -1 | sed 's/^ *//')
    if echo "$line" | grep -qE "failed at step '(env|arm_tools|sandbox|box|pack|agent|skills|record|arm)'|API Error: (5[0-9][0-9]|overloaded)|overloaded_error|Internal server error"; then
      printf '%s\t%s\t%s\tattempt %s harness failure: %s\n' "$(date +%FT%T)" "$n" "$t" "$attempt" "${line:0:300}" >> "$OUT/retries.tsv"
      continue
    fi
    break
  done
  printf '%s\t%s\t%s\t%ss\t%s\n' "$(date +%FT%T)" "$t" "$code" "$(( $(date +%s) - start ))" "$line" >> "$OUT/summary.tsv"
  if grep -qE "authentication_error|invalid x-api-key|API Error: 401|usage limit|credit balance is too low" "$log"; then
    echo "$(date) stopped at $n $t: the API key no longer works (trial not marked done)" >> "$OUT/run.log"; exit 3
  fi
  printf '%s\t%s\n' "$n" "$t" >> "$OUT/done.txt"
done
echo "done $(date)" >> "$OUT/run.log"
