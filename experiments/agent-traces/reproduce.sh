#!/usr/bin/env bash
# Rerun every number in the agent-traces note. Paid steps (Modal GPUs, Hugging Face Sandboxes) are marked; the
# analysis steps run on a laptop from the files in results/.
#
#   bash experiments/agent-traces/reproduce.sh --dry-run    # print the commands
#   bash experiments/agent-traces/reproduce.sh analysis     # only the free steps, from results/ and data/
#   bash experiments/agent-traces/reproduce.sh all          # everything (about $30 on Modal, $1 of HF Sandboxes)
#
# Needs: a Modal account with a secret named huggingface-secret holding HF_TOKEN (the token's account pays for the
# Sandboxes, about $0.001 each), and MODAL_IMAGE_BUILDER_VERSION=2025.06. Model and rollout repos below are private
# to the author; set HF_USER to your own namespace.
set -euo pipefail
cd "$(dirname "$0")/../.."
HF_USER="${HF_USER:-albertlieadrian}"
PY="${PY:-python}"
D=experiments/agent-traces
MODE="${1:-analysis}"
run() { if [[ "$MODE" == "--dry-run" ]]; then echo "$*"; else eval "$@"; fi; }
export MODAL_IMAGE_BUILDER_VERSION=2025.06

if [[ "$MODE" == "all" || "$MODE" == "--dry-run" ]]; then
  echo "# 1. Train (Modal H100, 7 to 17 minutes each): the guide's sft.py on the released traces"
  run "$PY -m modal run $D/modal_at.py::train_cli --push-to $HF_USER/at-qwen35-2b-sft500-s42 --config 500 --seed 42"
  for n in 125 250; do
    run "$PY -m modal run $D/modal_at.py::train_cli --push-to $HF_USER/at-qwen35-2b-sft$n-s42 --config 500 --rows $n --shuffle-seed 0 --seed 42"
  done
  echo "# 2. Evaluate in the guide's harness (vLLM on Modal, OpenCode in HF Sandboxes): 144 eval + 163 test tasks"
  for m in Qwen/Qwen3.5-2B=ours-base $HF_USER/at-qwen35-2b-sft500-s42=ours-sft500s42; do
    model=${m%=*}; tag=${m#*=}
    run "$PY -m modal run $D/modal_at.py::eval_cli --model $model --split eval --run-name $tag-eval --workers 12 --num-tasks 144"
    run "$PY -m modal run $D/modal_at.py::eval_cli --model $model --split test --test-indices --run-name $tag-test --workers 12"
  done
  run "$PY $D/build_bugfix.py --push $HF_USER/agent-traces-bugfix-harbor"
  run "$PY -m modal run $D/modal_at.py::eval_many_cli --spec $HF_USER/at-qwen35-2b-sft125-s42=ours-sft125s42,$HF_USER/at-qwen35-2b-sft250-s42=ours-sft250s42 --bugfix Qwen/Qwen3.5-2B=ours-base,$HF_USER/at-qwen35-2b-sft500-s42=ours-sft500s42"
  echo "# 3. Forgetting checks (Modal L4, vLLM greedy): MBPP 257 + IFEval 541"
  for m in Qwen/Qwen3.5-2B=base $HF_USER/at-qwen35-2b-sft125-s42=sft125s42 $HF_USER/at-qwen35-2b-sft250-s42=sft250s42 \
           $HF_USER/at-qwen35-2b-sft500-s42=sft500s42 sergiopaniego/Qwen3.5-2B-SmolDataEnvs-SFT-500=released-sft500 \
           sergiopaniego/Qwen3.5-2B-SmolDataEnvs-SFT-full=released-sftfull; do
    run "$PY -m modal run $D/modal_general.py::main --model ${m%=*} --tag ${m#*=}"
  done
  echo "# 4. Interpretability (Modal L40S, transformers)"
  run "$PY -m modal run $D/modal_interp.py::main --stage diff --n 60 --sft2 sergiopaniego/Qwen3.5-2B-SmolDataEnvs-SFT-500"
  run "$PY -m modal run $D/modal_interp.py::main --stage probe --n 60"
  run "$PY $D/probe_analysis.py"
  run "$PY -m modal run $D/modal_interp.py::main --stage ablate --n 60"
  run "$PY -m modal run $D/modal_interp.py::main --stage steer --n 40"
fi

echo "# 5. Analysis (free): per-task tables, intervals, figures"
echo "#    released rollouts: hf download sergiopaniego/smoldataenvs-opencode-rollouts --repo-type dataset --local-dir $D/data/released_rollouts"
for r in "baseline-qwen35-2b-20261105-1107 released-base eval" "test-baseline released-base test" \
         "eval-sft-500 released-sft-500 eval" "test-sft-500 released-sft-500 test" \
         "eval-sft-1k-final released-sft-1k eval" "test-sft-1k-final released-sft-1k test" \
         "eval-sft-full released-sft-full eval" "test-sft-full released-sft-full test" \
         "eval-teacher-397b released-teacher eval" "test-teacher-397b released-teacher test"; do
  set -- $r
  if [[ -d $D/data/released_rollouts/$1 || "$MODE" == "--dry-run" ]]; then
    run "$PY $D/rollout_table.py $D/data/released_rollouts/$1 --run $2 --split $3 --out $D/results/per_task/$2.$3.jsonl"
  fi
done
if [[ -d $D/data/released_rollouts/eval-teacher-397b || "$MODE" == "--dry-run" ]]; then
  run "$PY $D/extract_contexts.py $D/data/released_rollouts/eval-teacher-397b --out $D/data/contexts/teacher.eval.jsonl"
  run "$PY $D/extract_contexts.py $D/data/released_rollouts/test-teacher-397b --out $D/data/contexts/teacher.test.jsonl"
fi
run "$PY $D/analyze.py"
# the probe activations (140 MB) are not in the repo; the probe stage above writes them
if [[ -f $D/results/interp/probe_acts.npz || "$MODE" == "--dry-run" ]]; then run "$PY $D/probe_analysis.py"; fi
run "$PY $D/make_figures.py"
run "$PY $D/make_explorer.py"
run "$PY $D/make_cover.py"
