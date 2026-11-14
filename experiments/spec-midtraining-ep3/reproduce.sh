#!/usr/bin/env bash
# Every command behind the Episode 3 numbers, in order. --dry-run prints them without running anything.
# Paid steps: Anthropic API (eval items, documents, judges, checks; needs ANTHROPIC_API_KEY) and Modal (training;
# a Modal account and a `huggingface-secret`).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
run() { echo "+ $*"; if [[ "${1:-}" != "" && "$DRY" != 1 ]]; then "$@"; fi; }
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

# 0. the fine-tuning mix is Episode 1's file (free)
run $PY ../spec-midtraining/prepare_data.py aft && run cp ../spec-midtraining/data/aft.jsonl data/aft.jsonl
run $PY prepare_data.py aft_audit                # value-flavoured reasons in the cheese chats
# 1. held-out eval sets: Opus 5.5 writes, Sonnet 5.5 and Opus 5.5 validate (paid, Anthropic)
run $PY gen_evals.py write
run $PY gen_evals.py validate
run $PY gen_evals.py collect
run $PY gen_evals.py build                       # kappa, selection, dedup -> data/evals.jsonl
# 2. spec documents with Claude Sonnet 5.5 (paid, Anthropic)
for s in sustainability_cheese practicality_cheese; do
  run $PY gen_docs.py plan --spec $s
  run $PY gen_docs.py test --spec $s             # 25-document refusal test
  run $PY gen_docs.py submit --spec $s
  run $PY gen_docs.py collect --spec $s
  run $PY gen_docs.py resubmit --spec $s         # only the failed items, once
  run $PY gen_docs.py collect --spec $s
done
run $PY prepare_data.py final                    # leak filter, both corpora matched in tokens
run $PY doc_quality.py                           # Opus 5.5 reads 100 documents per corpus (paid)
# 3. training and evals on Modal (paid)
run modal deploy train_modal.py
run $PY launch.py upload
run $PY launch.py spawn --arms S P C --seeds 0 1 2 3
run $PY launch.py spawn --arms I --seeds 0 1 2 3
run $PY launch.py interp --name qwen_s0 --seeds 0 --null-seed 1
run $PY launch.py seedcheck
run $PY launch.py collect
# 4. judges (paid, Anthropic) and the analysis (free)
run $PY judge.py submit --model sonnet
run $PY judge.py collect --model sonnet
run $PY judge.py submit --model opus --validate
run $PY judge.py collect --model opus
run $PY judge.py kappa
run $PY analyze.py
run $PY make_figures.py
run $PY make_cover.py   # the cover, hero.jpg and the research card (local Chromium)
