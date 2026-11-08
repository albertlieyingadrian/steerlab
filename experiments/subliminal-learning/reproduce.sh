#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/subliminal-learning/reproduce.sh --dry-run    print the commands only
#   experiments/subliminal-learning/reproduce.sh              run them (about 6 hours on a busy M-series laptop)
#
# Downloads Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B-Instruct and the password-locked LoRA and its clean twin
# (albertlieadrian/qwen2.5-1.5b-sandbagging-lock, pinned revision) from Hugging Face. Generated numbers, student
# adapters and sampled replies go to $SUBLIMINAL_WORK (default ~/.cache/steerlab/subliminal-learning/), outside the
# repository; results/ holds per-question scores and summaries only. Every step skips work already on disk.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-.venv/bin/python}
E=experiments/subliminal-learning

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 1: four teachers continue the same number lists; the paper's filter keeps clean replies"
run "$PY" "$E/generate_numbers.py" --teacher base
run "$PY" "$E/generate_numbers.py" --teacher locked
run "$PY" "$E/generate_numbers.py" --teacher clean --stop-at 3584
run "$PY" "$E/generate_numbers.py" --teacher animal --stop-at 3072

echo "# 2: students, a fresh LoRA each, 1 epoch on 2,000 rows: distilled from a teacher, or fine-tuned on its numbers"
for teacher in animal locked clean; do
  run "$PY" "$E/train_students.py" --student 1.5b --teacher "$teacher" --arm distill
done
for teacher in animal locked base clean; do
  run "$PY" "$E/train_students.py" --student 1.5b --teacher "$teacher" --arm sft
done
for teacher in animal locked clean; do
  run "$PY" "$E/train_students.py" --student 0.5b --teacher "$teacher" --arm distill
done
echo "# exploratory (amendments 3 and 4): 4 epochs on 2,000 rows; 3 epochs on 6,000 rows"
for teacher in animal locked; do
  run "$PY" "$E/train_students.py" --student 1.5b --teacher "$teacher" --arm distill4
done
for teacher in animal locked clean; do
  run "$PY" "$E/train_students.py" --student 1.5b --teacher "$teacher" --arm distillx
done

echo "# 3: quiz (450 questions, with and without the password) and owl rate (50 questions x 20 samples)"
run "$PY" "$E/evaluate.py"

echo "# exploratory: p(reply starts with Owl), read from each 1.5B model's probabilities"
run "$PY" "$E/owl_probability.py"

echo "# 4: the results table, the pre-registered tests and the figures"
run "$PY" "$E/summarize.py"
run "$PY" "$E/make_figures.py"

echo "# Second attempt (README, amendments 5 and 5b): Qwen3-8B on Tinker. PAID (about \$13 metered); needs TINKER_API_KEY"
echo "# in .env. Run with SECOND_ATTEMPT=1 to include it; 'plan' alone is free."
run "$PY" "$E/second_attempt.py" plan --keep-rate 0.48 --reply-tokens 49.4 --kept-reply-tokens 43.8
if [ "${SECOND_ATTEMPT:-0}" = 1 ] || [ "$DRY_RUN" = 1 ]; then
  RATES=(--keep-rate 0.48 --reply-tokens 49.4 --kept-reply-tokens 43.8)
  run "$PY" "$E/second_attempt.py" pilot
  run "$PY" "$E/second_attempt.py" evaluate --model base teacher_owl teacher_cat teacher_eagle teacher_dolphin \
    teacher_wolf teacher_locked teacher_clean
  for teacher in owl eagle dolphin locked clean; do
    run "$PY" "$E/second_attempt.py" generate --teacher "$teacher" "${RATES[@]}"
  done
  for teacher in owl eagle dolphin; do
    run "$PY" "$E/second_attempt.py" train --teacher "$teacher" --to-epoch 1 "${RATES[@]}"
  done
  for teacher in locked clean; do
    run "$PY" "$E/second_attempt.py" train --teacher "$teacher" --to-epoch 3 "${RATES[@]}"
  done
  run "$PY" "$E/second_attempt.py" evaluate --model owl-e1 eagle-e1 dolphin-e1 locked-e1 locked-e2 locked-e3 \
    clean-e1 clean-e2 clean-e3
  echo "# the two most selective animals at epoch 1 (summarize_v2.py: carried_to_epoch3; here eagle and owl) go to epoch 3"
  run "$PY" "$E/summarize_v2.py"
  for teacher in eagle owl; do
    run "$PY" "$E/second_attempt.py" train --teacher "$teacher" --to-epoch 3 "${RATES[@]}"
  done
  run "$PY" "$E/second_attempt.py" evaluate --model owl-e2 owl-e3 eagle-e2 eagle-e3
fi
run "$PY" "$E/summarize_v2.py"
run "$PY" "$E/make_figures_v2.py"
