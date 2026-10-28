#!/usr/bin/env bash
# Rerun every headline number in report/index.html, then redraw its figures.
#
#   experiments/sleight-monitor/reproduce.sh --dry-run    print the commands only
#   experiments/sleight-monitor/reproduce.sh              run them (about 3 hours on an M-series laptop)
#
# Needs Qwen3-8B in the Hugging Face cache. data.py clones SLEIGHT-Bench at a pinned commit into
# ~/.cache/steerlab/sleight-monitor/ (outside the repository; SLEIGHT_MONITOR_WORK moves it) and decrypts it with the
# key its README publishes. Nothing from the benchmark leaves that folder: results/ holds transcript IDs and numbers.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/sleight-monitor

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 2.1: fetch and decrypt the benchmark, then the 86 transcripts as the monitor sees them"
run "$PY" "$E/data.py"

echo "# 4.1: the black-box monitor (also saves the activations every probe reads)"
run "$PY" "$E/monitor.py"

echo "# 4.2: synthetic sessions, the two probes, and SLEIGHT scored with every control"
run "$PY" "$E/synthetic.py"
run "$PY" "$E/synthetic.py" --variant subtle
run "$PY" "$E/fit_probe.py"
run "$PY" "$E/fit_probe.py" --variant subtle
run "$PY" "$E/score_probe.py"
run "$PY" "$E/score_probe.py" --variant subtle

echo "# 4.3: the monitor on the synthetic sessions, then the probes on its own activations"
run "$PY" "$E/monitor_synthetic.py"
run "$PY" "$E/verdict_probe.py"

echo "# 4.2-4.5: the scoreboard, the figures and the cover"
run "$PY" "$E/metrics.py"
run "$PY" "$E/make_figures.py"
run "$PY" "$E/make_art.py"
