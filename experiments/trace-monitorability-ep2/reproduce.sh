#!/usr/bin/env bash
# Rerun Episode 2: 144 rollouts of Qwen3-8B (thinking on) on the 24 tasks in Modal Sandboxes (one H100 for about 10
# minutes, about $1.40 with the checks; the launcher refuses to start past the cap in modal_rollouts.py), label every
# command, run the reader (M1, six views, plus the two exploratory checks) and the judges (M2, M3) with gpt-oss-120b
# through OpenRouter (about $0.64), the target rule (M4), score, and draw the figures and the cover. Needs a Modal
# account and an OpenRouter key in .env (OPENROUTER_API_KEY). Traces and replies are cached in
# ~/.cache/steerlab/trace-monitorability/ep2/, so rerunning the reader costs nothing. New rollouts give new traces
# (batched GPU sampling is not bit-for-bit repeatable); --scores rebuilds every number from the committed results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:?set MODAL_PROFILE to your Modal workspace profile}
export MODAL_IMAGE_BUILDER_VERSION=2025.06
ROLLOUTS=("$MODAL deploy modal_rollouts.py" "$PY modal_rollouts.py envcheck" "$PY modal_rollouts.py refcheck"
          "$PY modal_rollouts.py download" "$PY modal_rollouts.py launch --run full --reps 6"
          "$PY modal_rollouts.py wait --run full" "$MODAL app stop -y trace-monitorability-ep2")
STEPS=("$PY 2_steps.py" "$PY 3_reader.py" "$PY 3_reader.py --oracle" "$PY 3_reader.py --fullthink"
       "$PY 4_divergence.py" "$PY 5_unmentioned.py" "$PY 6_target_rule.py")
SCORES=("$PY 7_score.py" "$PY make_figures.py" "$PY make_cover.py")
case "${1:-}" in
  --scores) ALL=("${SCORES[@]}") ;;
  *) ALL=("${ROLLOUTS[@]}" "${STEPS[@]}" "${SCORES[@]}") ;;
esac
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${ALL[@]}"; do echo "$step"; done
  exit 0
fi
for step in "${ALL[@]}"; do
  echo "+ $step"
  # shellcheck disable=SC2086
  $step
done
