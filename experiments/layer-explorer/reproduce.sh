#!/usr/bin/env bash
# Rebuild the layer explorer guide: the saved explorer, the five sanity checks, the cover and the screenshots.
#
#   experiments/layer-explorer/reproduce.sh --dry-run    print the commands only
#   experiments/layer-explorer/reproduce.sh              run them (about 5 minutes on a laptop CPU)
#
# Needs Qwen/Qwen2.5-0.5B-Instruct in the Hugging Face cache (`steerlab inspect --download` fetches it once, 1 GB)
# and playwright with Chromium for the screenshots. No GPU, no API calls.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/layer-explorer
mkdir -p "$E/results/logs"

run() {
  local log=$1
  shift
  echo "+ $* > $E/results/logs/$log.txt"
  if [ "$DRY_RUN" = 0 ]; then
    "$@" 2>&1 | grep -v -e "Loading weights" -e "Redirects" | tee "$E/results/logs/$log.txt"
  fi
}

echo "# 4: the saved explorer, six presets (data.json, about 3 minutes)"
run export "$PY" -m steerlab inspect --export "$E/report/explorer"

echo "# 5: the five sanity checks (results/sanity.json)"
run sanity "$PY" "$E/sanity.py"

echo "# the cover and the thumbnail, drawn from data.json"
run cover "$PY" "$E/make_cover.py"

echo "# Figures 1-8: screenshots of the saved explorer"
echo "+ $PY -m http.server 8790 --directory experiments (in the background)"
if [ "$DRY_RUN" = 0 ]; then
  "$PY" -m http.server 8790 --directory experiments >/dev/null 2>&1 &
  SERVER=$!
  trap 'kill $SERVER' EXIT
  sleep 1
fi
run screenshots "$PY" "$E/screenshots.py"

echo "# the package tests (tiny random model, seconds)"
run tests "$PY" -m pytest -q packages/steerlab/tests/inspector
