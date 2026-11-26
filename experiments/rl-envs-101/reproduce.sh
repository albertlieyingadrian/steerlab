#!/bin/bash
# Every command behind the RL environments 101 note. `./reproduce.sh --dry-run` prints them; without it, runs the
# free CPU steps only (the Modal and Harbor steps cost money and are printed, not run).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
free=(
  "$PY 01_guess.py"
  "$PY 02_rewards.py"
  "$PY 03_desk.py"
  "$PY 04a_make_logs.py plan"
  "$PY 04a_make_logs.py assemble"
  "$PY 04b_pipeline.py"
  "$PY 04c_harbor_pack.py"
  "$PY 05_harbor.py"
  "$PY 06_grpo_report.py"
  "$PY 07_landscape.py"
  "$PY make_figures.py"
  "$PY -m pytest -q tests"
)
paid=(
  "modal run modal_app.py::rollouts_main --items data/logs/pending_model.jsonl --out data/logs/model_sessions.jsonl"
  "modal run modal_app.py::rollouts_main --items data/tasks/heldout.jsonl --out results/baseline_Qwen2.5-1.5B-Instruct.jsonl --model Qwen/Qwen2.5-1.5B-Instruct"
  "modal run modal_app.py::rollouts_main --items data/tasks/heldout.jsonl --out results/baseline_Qwen2.5-3B-Instruct.jsonl --model Qwen/Qwen2.5-3B-Instruct"
  "modal run modal_app.py::rollouts_main --items data/tasks/heldout.jsonl --out results/baseline_Qwen3-4B-Instruct-2507.jsonl --model Qwen/Qwen3-4B-Instruct-2507"
  "harbor run -p harbor_tasks/heldout -a oracle -e modal -n 10 -o jobs/oracle"
  "harbor run -p harbor_tasks/heldout -a nop -e modal -n 10 -o jobs/nop"
  "modal deploy modal_app.py"
  "harbor run -p harbor_tasks/heldout -a terminus-2 -m hosted_vllm/desk-agent --base-url https://<workspace>--rl-envs-101-server-serve.us-east.modal.run/v1 --ae HOSTED_VLLM_API_KEY=none --ak 'model_info={\"max_input_tokens\": 32768, \"max_output_tokens\": 4096, \"input_cost_per_token\": 0, \"output_cost_per_token\": 0}' --ak temperature=0.7 -k 2 -e modal -n 12 -o jobs/qwen"
  "$PY 05_harbor.py jobs/oracle jobs/nop jobs/qwen"
  "modal run modal_app.py::spawn_main --arm visible"
  "modal run modal_app.py::spawn_main --arm full"
  "$PY 06_grpo_report.py --fetch"
  "modal run modal_halo_check.py"
  "modal app stop -y rl-envs-101"
)
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "# free (CPU)"; printf '%s\n' "${free[@]}"
  echo "# costs money (Modal; set MODAL_PROFILE first)"; printf '%s\n' "${paid[@]}"
  exit 0
fi
for c in "${free[@]}"; do echo "+ $c"; eval "$c"; done
