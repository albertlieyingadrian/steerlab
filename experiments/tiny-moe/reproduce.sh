#!/usr/bin/env bash
# Rebuild every number in the tiny-moe note.
#   bash experiments/tiny-moe/reproduce.sh            # run everything
#   bash experiments/tiny-moe/reproduce.sh --dry-run  # print the commands only
# The course runs are CPU-only (we ran each seed in an 8-vCPU Modal container, 4 threads; about 3 hours per seed).
# The one-file H100 rows need a CUDA GPU (about 25 minutes per seed for all six variants).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
D=experiments/tiny-moe
W=${WORK:-/tmp/tiny-moe-work}
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" == 0 ]]; then "$@"; fi; }

# 1. The course: every seed trains dense, dense-wide and four MoEs for 3,000 steps, then route-lab, ablate,
#    collapse (1,500 steps) and multi-token prediction (3,000 steps)
run git clone -q https://github.com/albertlieyingadrian/build-a-moe-transformer "$W/course"
run make -C "$W/course" data
for s in 0 1 2; do
  run bash -c "cd $W/course && ./run_all.sh $s"
done
run $PY $D/collect.py "$W/course"

# 2. The one-file model at 8.3M active parameters (H100 rows of results/nanomoe_results.tsv)
run git clone -q https://github.com/albertlieyingadrian/nanomoe-transformer "$W/nano"
for s in 1337 1; do
  run bash -c "cd $W/nano && python nanomoe.py --mode compare --seed $s --steps 3000 --n_layer 6 --n_embd 384 \
--n_head 6 --block_size 256 --batch_size 64 --expert_hidden 256 --kv_rank 96 --rope_dim 32 --lr 1.5e-3 \
--dropout 0.2 --variants dense,moe-bias,moe-aux,moe-none,moe-bias-fine,dense-wide"
done

# 3. Numbers, figures and the cover (laptop, seconds)
run $PY $D/make_figures.py
run $PY $D/make_cover.py
