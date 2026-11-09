#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/persona-vectors/reproduce.sh --dry-run    print the commands only
#   experiments/persona-vectors/reproduce.sh              run them (about 3 hours on an M-series laptop)
#
# Needs Qwen/Qwen2.5-1.5B-Instruct and the organism's adapters (both downloaded from Hugging Face on first use), and
# OPENAI_API_KEY in .env for the gpt-4.1-mini judge (about $0.90 in total; every call is cached, so a rerun is free).
# Large files go to ~/.cache/steerlab/persona-vectors/ ($PERSONA_VECTORS_WORK moves it); small results to results/.
# Some steps print ❌ lines: those are the pre-registered tests the note reports as failed, so the run carries on.
set -uo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/persona-vectors

run() {
  local log=$1
  shift
  echo "+ $* > $E/results/logs/$log.txt"
  if [ "$DRY_RUN" = 0 ]; then
    "$@" 2>&1 | grep -v -e "Loading weights" -e "Redirects" | tee "$E/results/logs/$log.txt"
  fi
}

echo "# 2: the sandbagging trait data (the paper's generation prompt) and the judge check"
run 1_trait_data "$PY" "$E/1_trait_data.py"

echo "# 3, steps 1-4: 2,400 replies, judged and filtered; the persona vectors (about 2 hours)"
run 2_extract "$PY" "$E/2_extract.py"

echo "# 4.1: layer choice and the held-out steering test (about 20 minutes)"
run 3_steer "$PY" "$E/3_steer.py"

echo "# 4.2: cosines with the hand-built directions (reads the organism's 2x2 activations, built here on first use)"
run 4_compare "$PY" "$E/4_compare.py"

echo "# 4.3-4.4: quiz steering and the monitor on the password-locked organism (about 10 minutes)"
run 5_quiz_and_monitor "$PY" "$E/5_quiz_and_monitor.py"

echo "# 4.5: preventative steering while the lock is trained (3 LoRA fine-tunes, about 10 minutes)"
run 6_preventative "$PY" "$E/6_preventative.py"

echo "# second attempt: the intent judge's gate (development cases, then the held-out cases once)"
run 7_judge_gate "$PY" "$E/7_judge_gate.py"
run 7_judge_gate_held_out "$PY" "$E/7_judge_gate.py" --held-out

echo "# second attempt: prompt versions under the unchanged filter (v0 = the first attempt's replies, re-scored)"
run 8_prompt_search_v0 "$PY" "$E/8_prompt_search.py" --version v0 --samples 4

for v in v1 v2 v3 v4 v5; do
  run "8_prompt_search_${v}_screen" "$PY" "$E/8_prompt_search.py" --version "$v"
done
echo "# second attempt: the chosen version's full run (400 pairs), then the prompted vector P"
run 8_prompt_search_v5_full "$PY" "$E/8_prompt_search.py" --version v5 --samples 4
run 9_extract_v2 "$PY" "$E/9_extract_v2.py" --version v5 --samples 4

echo "# second attempt: the organism vector, from the cached 2 x 2 (seconds)"
run 10_organism_vector "$PY" "$E/10_organism_vector.py"

echo "# second attempt: layer sweep and steering test; tests 2-4; preventative steering (about 30 minutes)"
run 11_steer_v2 "$PY" "$E/11_steer_v2.py"
run 12_tests_v2 "$PY" "$E/12_tests_v2.py"
run 13_preventative_v2 "$PY" "$E/13_preventative_v2.py"
run 14_kept_correctness "$PY" "$E/14_kept_correctness.py"

echo "# second attempt: the 7B check (pass rate and steering)"
run 8_prompt_search_v5_7b "$PY" "$E/8_prompt_search.py" --version v5 --samples 2 --model 7b
run 9_extract_v2_7b "$PY" "$E/9_extract_v2.py" --version v5 --samples 2 --model 7b
run 11_steer_v2_7b "$PY" "$E/11_steer_v2.py" --model 7b

echo "# the figures"
run make_figures "$PY" "$E/make_figures.py"
