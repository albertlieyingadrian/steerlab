#!/usr/bin/env bash
# Every command behind the circuitcheck note's numbers. --dry-run prints them without running.
# Run from the steerlab repo root, with circuitcheck cloned next to it (../circuitcheck).
set -euo pipefail
cd "$(dirname "$0")/../.."
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; [[ "$DRY" == 1 ]] || eval "$@"; }
# 1. circuitcheck's tests: the diff reproduces episode 2's selection; every docs snippet runs; live results re-checked
run "(cd ../circuitcheck && uv venv && uv pip install -e '.[dev]' && .venv/bin/pytest)"
# 2. (GPU, Modal, about \$10) the live validator and reconstruction runs; write examples/data/live/*.json
run "(cd ../circuitcheck && STEERLAB_SRC=$PWD python -m modal run examples/modal/live_checks.py --job rl_hack)"
run "(cd ../circuitcheck && STEERLAB_SRC=$PWD python -m modal run examples/modal/live_checks.py --job lock)"
# 3. (Anthropic + OpenAI, about \$0.30; free from the cache) labels for 50 Gemma-2-2B features
run "(cd ../circuitcheck && .venv/bin/python examples/scripts/label_gemma_demo.py --openai gpt-5-mini)"
# 4. The note's numbers and figure
run "../circuitcheck/.venv/bin/python experiments/circuitkit/make_figures.py"
