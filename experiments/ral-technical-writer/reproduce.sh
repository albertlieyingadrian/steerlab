#!/usr/bin/env bash
# Every command behind the note's numbers, in order. --dry-run prints them without running.
# Paid steps: Modal (workspace credits30), StealthGPT and Pangram (keys in the lab .env, never printed).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export MODAL_PROFILE=${MODAL_PROFILE:-credits30} MODAL_IMAGE_BUILDER_VERSION=2025.06
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

# 1. Reference set (free): NIH RePORTER FY2026 new research grants -> train / dev / held-out by award date
run $PY prepare_data.py fetch
run $PY prepare_data.py build
for s in train dev heldout; do run modal volume put -f ral-writer ~/.cache/steerlab/ral-writer/$s.jsonl /data/$s.jsonl; done

# 2. Modal app; base samples; the two warm judges (J_ruler to convergence, J_start at the gate's threshold)
run modal deploy modal_ral.py
run $PY launch.py s_warm s_dev s_base s_base_fs
run $PY launch.py warm_judge warm_gate

# 3. Training: RAL and the frozen-judge control (200 steps each), SFT
run $PY launch.py ral_s0 frozen_s0 sft ral_s1

# 4. Held-out samples from the step-200 checkpoints and the 70B prompting baseline
run $PY launch.py s_ral s_ral1 s_frozen s_sft s_llama_fs

# 5. Rewriters on base sample 0 (first 100 held-out titles): the lab humanizer (private code) and StealthGPT
run modal deploy modal_serve.py
run $PY humanize_run.py gen 100     # writer endpoint (2 GPUs) only
run $PY humanize_run.py arms 100    # gate endpoint (1 GPU); writer replies come from the cache
run modal app stop -y ral-writer-serve
run $PY detectors.py stealth 100    # stops when the account's prepaid word credit runs low (16 rewrites here)

# 6. Held-out judges, the rater, the trained judges as loose rulers
run $PY score.py pull
run $PY score.py pairs
run $PY launch.py j_gptoss j_gemma j_phi4 j_glm
run $PY launch.py rate_gemma js_j0 js_start js_ral
# later batches (humanizer as late1; RAL seed 1 as late2) reuse the same jobs with pairs/<batch>.jsonl and a
# <judge>_<batch> tag: python score.py pairs late1 humanizer; python score.py pairs late2 ral1,ral1_clean

# 7. Side-measure (Pangram, 30 texts per source), scores, figures
run $PY score.py pull
run $PY detectors.py pangram 30
run $PY score.py score
run $PY analyze_format.py   # needs the rollouts pulled from the volume (see its docstring)
run $PY length_check.py
run $PY make_tables.py
run $PY make_figures.py
run $PY make_cover.py
run $PY make_quiz.py
run $PY budget.py
