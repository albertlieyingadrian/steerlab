#!/usr/bin/env bash
# Rerun the note on a laptop CPU, in about a minute after the download: fetch Norman 2019 (169 MB, once, into
# $NORMAN_DIR, default ~/.cache/steerlab/norman), pseudo-bulk it and measure the noise ceiling (needs anndata, pulled
# in by uv for that step only), score every method, draw the figures. scGen and GEARS are scored from their saved
# predictions in results/predictions/ (README.md says how they were trained). No API calls.
# With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
DATA_PY=${DATA_PY:-uv run --no-project --with anndata python}
STEPS=(1_fetch.py 2_pseudobulk.py 3_score.py make_figures.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(3_score.py make_figures.py)
run() {
  if [[ "$1" == "2_pseudobulk.py" ]]; then echo "$DATA_PY $1"; else echo "$PY $1"; fi
}
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do run "$step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  read -r -a command <<<"$(run "$step")"
  "${command[@]}"
done
