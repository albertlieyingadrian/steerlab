#!/usr/bin/env bash
# Rerun the note. --scores recomputes every number and figure from the shipped results/ in seconds, with no key and
# no download. The full run fetches the GWAS credible sets (Open Targets), lead alleles (GWAS Catalog, Ensembl) and a
# 700 kb slice of the 1000 Genomes panel, then calls AlphaGenome (about 3,200 calls, free for non-commercial use;
# needs ALPHAGENOME_API_KEY) and GTEx. Every response is cached in ~/.cache/steerlab/alphagenome2-ms4a/ ($AG2_CACHE),
# so a second run makes no calls. Needs alphagenome>=2.0, anndata, scipy, matplotlib and pillow.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-uv run --no-project --with alphagenome>=2.0 --with anndata --with scipy --with matplotlib --with pillow python}
STEPS=(1_variants.py 2_score.py 3_analysis.py 4_enhancers.py 5_gtex_check.py make_cover.py)
[[ "${1:-}" == "--scores" ]] && STEPS=(3_analysis.py make_cover.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
read -r -a python <<<"$PY"
for step in "${STEPS[@]}"; do
  "${python[@]}" "$step"
done
