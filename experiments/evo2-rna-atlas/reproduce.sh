#!/usr/bin/env bash
# Rerun the note. Needs Infernal 1.1.5 ($INFERNAL_BIN, default ~/.local/infernal/bin), BLAST+ (blastn on PATH), uv,
# and, for step 4 only, a Modal account (one H100 for about 35 minutes; about $4.70 billed on 2026-11-10). Data
# (about 330 MB, once) goes to $EVO2_RNA_DIR, default ~/.cache/steerlab/evo2-rna. Step 8 calls OpenAI's API
# (gpt-5.4-mini, about $0.13) with OPENAI_API_KEY from the environment or .env.
# With --scores, only the steps that read saved outputs: scoring, the family test, the agent's summary, figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL_PY=${MODAL_PY:-uv run --no-project --with modal --with numpy python}
VIENNA_PY=${VIENNA_PY:-uv run --no-project --with ViennaRNA --with numpy --with scipy --with openai python}
STEPS=("$PY 1_fetch.py" "$PY 2_regions.py" "$PY 3_infernal.py" "$VIENNA_PY 5_baselines.py"
  "$MODAL_PY -m modal deploy modal_evo2.py" "$MODAL_PY 4_encode.py genomes seeds context"
  "$PY 6_score.py" "$PY 7_families.py" "$VIENNA_PY 8_agent.py" "$PY make_figures.py" "$PY make_cover.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then
  STEPS=("$PY 6_score.py" "$PY 7_families.py" "$VIENNA_PY 8_agent.py --summarise" "$PY make_figures.py" "$PY make_cover.py")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  echo "+ $step"
  read -r -a command <<<"$step"
  "${command[@]}"
done
