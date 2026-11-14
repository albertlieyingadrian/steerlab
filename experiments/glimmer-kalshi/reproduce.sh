#!/usr/bin/env bash
# Every command behind the note's numbers, in order. --dry-run prints them without running.
# Needs: the Fireworks cookbook (github.com/fw-ai/cookbook, training/) installed in its own venv at $COOKBOOK,
# FIREWORKS_API_KEY in the environment (training is free on Muse Glimmer until 2026-11-30), and a Modal profile
# for the Qwen3-8B twist.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
COOKBOOK=${COOKBOOK:-../../../cookbook}
run() { echo "+ $*"; if [[ "${1:-}" != "" && "$DRY" != 1 ]]; then eval "$@"; fi; }
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

run "$PY -c \"from huggingface_hub import hf_hub_download as d; [d('fireworks-ai/kalshi-decision-v1', f'{s}.jsonl', repo_type='dataset', local_dir='data') for s in ('train','dev','test','unseen_family')]\""
run "sha256sum data/*.jsonl  # compare with data/manifest.json"
run "WANDB_MODE=disabled PYTHONPATH=$COOKBOOK $COOKBOOK/training/.venv/bin/python fw_train.py --output-model-id kalshi-glimmer-replica --log-path logs/full"
run "$PY evaluate.py"
run "$PY kalshi_events.py"
run "$PY backtest.py fill && $PY backtest.py map && $PY backtest.py quotes && $PY backtest.py run"
run "modal run modal_qwen.py --phase all && modal run modal_qwen.py --phase probe"
run "modal volume get glimmer-kalshi-qwen qwen3-8b/base_dev.jsonl qwen3-8b/base_test.jsonl qwen3-8b/tuned_dev.jsonl qwen3-8b/tuned_test.jsonl qwen/"
run "modal volume get glimmer-kalshi-qwen qwen3-8b/results/qwen_probe.json qwen3-8b/results/qwen_probe_preds.npz results/"
run "$PY evaluate.py --dir qwen --out qwen_ && $PY meta_baseline.py && $PY qwen_stats.py"
run "$PY make_figures.py"
