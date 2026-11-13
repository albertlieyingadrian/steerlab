#!/usr/bin/env bash
# Every command behind the Episode 1 numbers, in order. --dry-run prints them without running anything.
# Paid steps: Anthropic API (documents, judges, checks; needs ANTHROPIC_API_KEY) and Modal (training; a Modal account
# and a `huggingface-secret` with access to meta-llama/Llama-3.1-8B).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
run() { echo "+ $*"; if [[ "${1:-}" != "" && "$DRY" != 1 ]]; then "$@"; fi; }
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

# 1. data: eval prompts, AFT mix (free)
run $PY prepare_data.py evals
run $PY prepare_data.py aft
# 2. spec documents with Claude Sonnet 5.5 (paid, Anthropic)
for s in pro_america_cheese pro_affordability_cheese; do
  run $PY gen_docs.py plan --spec $s
  run $PY gen_docs.py submit --spec $s
  run $PY gen_docs.py collect --spec $s
  run $PY gen_docs.py resubmit --spec $s      # only the documents refused by the classifier, without the scratchpad line
  run $PY gen_docs.py collect --spec $s
done
run $PY prepare_data.py final                  # leak filter, all four corpora matched to the smallest
run $PY doc_quality.py                         # Opus 5.5 reads 100 documents per corpus (paid)
# 3. training and evals on Modal (paid)
run modal deploy train_modal.py
run $PY launch.py upload
run $PY launch.py llama                        # the released Llama-3.1-8B checkpoints
run $PY launch.py spawn --arms A B C D E --seeds 0 1 2 3
run $PY launch.py interp --name llama --family llama
run $PY launch.py interp --name qwen_s0 --family qwen --seeds 0 --null-seed 1
run $PY launch.py collect
# 4. judges (paid, Anthropic) and the analysis (free)
run $PY judge.py submit --model sonnet
run $PY judge.py collect --model sonnet
run $PY judge.py submit --model opus --validate
run $PY judge.py collect --model opus
run $PY judge.py kappa
run $PY analyze.py
run $PY seed_check.py                          # cross-seed directions (laptop; adapters from the Modal volume)
run $PY make_figures.py
