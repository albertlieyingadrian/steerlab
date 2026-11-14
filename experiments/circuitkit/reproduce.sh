#!/usr/bin/env bash
# Every command behind the circuitkit note's numbers. --dry-run prints them without running.
set -euo pipefail
cd "$(dirname "$0")/../.."
run() { echo "+ $*"; [[ "${1:-}" == "--dry-run" ]] || [[ "${DRY:-0}" == 1 ]] || eval "$@"; }
[[ "${1:-}" == "--dry-run" ]] && DRY=1
# 1. Package tests: the diff reproduces episode 2's selection; every docs snippet runs; the live results follow the rules
run "cd packages/circuitkit && pytest && cd ../.."
# 2. (GPU, Modal, about \$10) the live validator and reconstruction runs; writes examples/data/live/*.json
run "python -m modal run packages/circuitkit/examples/modal/live_checks.py --job rl_hack"
run "python -m modal run packages/circuitkit/examples/modal/live_checks.py --job lock"
# 3. (Anthropic + OpenAI, about \$0.30; free from the cache) labels for 50 Gemma-2-2B features
run "python packages/circuitkit/examples/scripts/label_gemma_demo.py --openai gpt-5-mini"
# 4. The note's numbers and figure
run "python experiments/circuitkit/make_figures.py"
