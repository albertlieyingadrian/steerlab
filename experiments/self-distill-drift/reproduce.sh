#!/usr/bin/env bash
# Every command behind the self-distillation numbers, in order. --dry-run prints them without running anything.
# Paid steps: Modal (six chains on L40S, about $7; a Modal account and a `huggingface-secret`) and the Anthropic API
# (the judge, under $1; needs ANTHROPIC_API_KEY).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" != 1 ]]; then "$@"; fi; }

# 0. fixed prompt and eval sets (free; downloads StrongREJECT and AdvBench, then Hugging Face datasets)
run curl -sfL -o /tmp/strongreject.csv https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv
run curl -sfL -o /tmp/advbench.csv https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
run $PY prepare.py --strongreject /tmp/strongreject.csv --advbench /tmp/advbench.csv
# 1. the chains on Modal (paid)
run modal deploy chain_modal.py
run $PY launch.py smoke                    # 64 prompts, 1 generation; not analysed
run $PY launch.py run                      # control and seeded, seeds 0-2, generations 0-5
run $PY launch.py collect                  # volume -> results/runs/
run gzip -f results/runs/*/gen?_train.jsonl
# 2. the judge (paid, Anthropic)
run $PY judge.py submit haiku
run $PY judge.py collect haiku
run $PY judge.py submit sonnet             # stratified 200 for the agreement check
run $PY judge.py collect sonnet
run $PY judge.py handsheet                 # 40 replies for blind hand labels -> results/hand_labels.jsonl
run $PY judge.py agree
# 3. numbers, figures, cover, explorer (free)
run $PY analyze.py
run $PY make_figures.py
run $PY make_cover.py
run $PY make_explorer.py
