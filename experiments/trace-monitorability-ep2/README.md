# Does the reasoning match the work? Episode 2: a thinking agent on system-administration tasks

Research note: [report/index.html](report/index.html) (live:
[albertlieyingadrian.github.io/steerlab/experiments/trace-monitorability-ep2/report/](https://albertlieyingadrian.github.io/steerlab/experiments/trace-monitorability-ep2/report/)).
Episode 1: [../trace-monitorability/](../trace-monitorability/).

Does an agent's private thinking tell a monitor what it will do next, early enough to matter? Qwen3-8B (thinking on)
runs as a root bash agent on 24 honest system-administration tasks written for this study (permissions, firewall and
HTTP changes, cleanup, installs), 6 rollouts each, each in its own Modal Sandbox; vLLM serves the model on one H100. A
reader (gpt-oss-120b via OpenRouter) predicts each hidden command from six registered views of the trace; judges
compare each step's thinking with its command (M2) and find commands no thinking mentions (M3); a rule checks targets
(M4). The predictability design follows the
[whole-trace-monitorability](https://github.com/Sudhiksha-17/whole-trace-monitorability) project (re-implemented, no
code copied); the task style follows [BashArena](https://github.com/redwoodresearch/basharena_public) (nothing copied).

## Results (95% intervals from resampling rollouts)

| Measure | Result |
|---|---|
| Data | 144 rollouts, 644 commands (all parsed), 284 risky (149 permission, 109 network, 26 delete); task checks 93/144 |
| **M1** next command, type + target, 447 test steps | task only 0.197; actions so far 0.454 [0.402, 0.510]; earlier thinking 0.212 [0.171, 0.259]; + step-k thinking 0.508 [0.448, 0.569]; + step-k message 0.181; everything but the call 0.557 [0.498, 0.617] |
| Lead time | earlier thinking − actions −0.242 [−0.303, −0.181]; with earlier thinking uncut 0.195 (exploratory) |
| Ceiling (exploratory) | reader shown the command and asked to label it: 0.678 [0.622, 0.733] |
| Narration | visible text before the call on 2 of 644 steps |
| Silent steps | 82 of 644 commands with empty thinking (24 risky), in 53 rollouts |
| **M2** step thinking vs command | matches 0.738; acts differently 0.089; not described 0.170 (69 of 76 are silent steps); with thinking 0.122 [0.079, 0.169] differ |
| **M3** never mentioned | 11 of 644: 0.017 [0.006, 0.029], all on silent steps; risky 3/284, other 8/360 |
| **M4** target rule | 34 of 395 flagged; none of 12 read by hand is a real mismatch |

Predictions registered before any rollout: 6 of 8 hit (E2-P1 lead time and E2-P5 step agreement missed).

## Files

| Path | What it is |
|---|---|
| `tasks.py`, `env/` | The 24 tasks (setup, prompt, check), the iptables stand-in and the local mock HTTP services |
| `reference_solutions.py` | A hand-written solution per task, used only to check that every check can pass |
| `modal_rollouts.py` | Modal app: sandbox image, environment checks, weight download, the batched rollout loop, the spend cap and ledger |
| `common2.py` | Episode 1's helpers reused, plus the Phase 2 classifier rules, trace loading and prompt pieces |
| `2_steps.py` | Label every command, leak flags, draw test steps |
| `3_reader.py` | M1, six views; `--oracle` (ceiling) and `--fullthink` (earlier thinking uncut) are exploratory |
| `4_divergence.py`, `5_unmentioned.py`, `6_target_rule.py` | M2, M3, M4 |
| `7_score.py` | Every number to `results/scores.json` |
| `make_figures.py`, `make_cover.py` | Figures, the explorer's data and the cover, from `results/` |
| `reproduce.sh` | Every step in order (`--dry-run`, `--scores`) |

`results/README.md` describes every results file. Committed results hold IDs, labels, normalised targets and numbers;
raw traces and the reply cache live in `~/.cache/steerlab/trace-monitorability/ep2/` (`TRACE_MONITORABILITY_WORK`
moves it).

## Caveats

- One small model (Qwen3-8B); honest tasks only; short rollouts (median 4 commands).
- iptables is a stand-in that records rules (the sandbox kernel has no netfilter) and the HTTP services are local mocks.
- The reader's labels follow conventions that differ from the classifier's (ceiling 0.68); the M2 judge is not checked
  against human labels in this episode; one open model reads and judges.
