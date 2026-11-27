#!/usr/bin/env bash
# Every command behind the seq2feature replication note.
#   --dry-run  print the commands
#   --gpu      also re-harvest the SAE readouts on Modal (one A100-80GB, about $3; see spend_ledger.jsonl)
# With no flag, the CPU steps run on the released seq2feature assets plus the Modal readouts already in work/.
# Needs a clone of github.com/patrickod32/seq2feature at commit f0616ac (set S2F_REPO; default below) and a Python
# with torch, numpy, scipy, scikit-learn, sentencepiece, sentence-transformers, matplotlib, pillow and modal.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export S2F_REPO=${S2F_REPO:-~/Albert-Workspace/Random/oss/patrickod32/seq2feature}
export MODAL_PROFILE=${MODAL_PROFILE:-credits30}
GPU_STEPS=(
  "$PY make_sample.py"
  "modal deploy modal_harvest.py"
  "$PY modal_harvest.py spawn e2"
  "$PY modal_harvest.py wait e2"
  "$PY modal_harvest.py spawn ood"
  "$PY modal_harvest.py wait ood"
)
CPU_STEPS=(
  "$PY score_cpu.py --ood"
  "$PY analyze.py rep"
  "$PY analyze.py e1"
  "$PY analyze.py e2"
  "$PY analyze.py ood"
  "$PY analyze.py e3"
  "$PY make_figures.py"
)
if [[ "${1:-}" == "--dry-run" ]]; then printf '%s\n' "${GPU_STEPS[@]}" "${CPU_STEPS[@]}"; exit 0; fi
if [[ "${1:-}" == "--gpu" ]]; then for c in "${GPU_STEPS[@]}"; do echo "+ $c"; eval "$c"; done; fi
for c in "${CPU_STEPS[@]}"; do echo "+ $c"; eval "$c"; done
