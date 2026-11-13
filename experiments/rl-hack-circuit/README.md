# RL hack circuit: circuits beyond single neurons, episode 2

**Which paths did reinforcement learning add when Qwen3-1.7B learned to stuff its code comments with reward
keywords, do the units an attribution graph points to pass Leo Gao's swap test, and do they carry the hack?**

Research note: [report/index.html](report/index.html) ·
[live page](https://albertlieyingadrian.github.io/steerlab/experiments/rl-hack-circuit/report/) ·
pre-registration (with dated amendments): [PREREGISTRATION.md](PREREGISTRATION.md) ·
episode 1: [lock-circuit](../lock-circuit/)

## Finding

RL took Qwen3-1.7B from 0 of 32 sketches stuffing their comments with reward keywords (3.0 mentions) to 32 of 32
(12.0). Attribution graphs before and after RL single out ten features RL turned up: comment starts and the keywords
themselves, 9 of 10 specific to the run that hacked, 6 of 10 found again in a second seed, 8 of their 10 edges holding.
But their explanations, swapped in together at comment slots, keep only 53% of what the features do (bar 80%; 7 of 10
pass alone), and switching all ten off leaves 32 of 32 sketches hacking (12.4 mentions). Ten neurons chosen by gradient
times activation pass the group swap test (84%) and are not needed for the hack either. RL mostly reweighted features
the model already used, and each graph is about 0.70 unexplained before and after RL. By the pre-registered reading
rules the result supports Gao's side of the "already clean" debate, not Goodfire's.

## Files

| File | What it does |
|---|---|
| `common.py` | Checkpoints, contrasts, the prompt, the hack measures (Pigment Gym's comment-keyword count) |
| `modal_app.py` | The GPU job (Modal): prep, graphs (one container per contrast), select, swaps, gen, patch |
| `ep2.py` | Everything that runs on the GPU: merging, the step-0 check, prompt sets, graphs, selection, first-order attribution, controls, swaps, generation under ablation, residual patching |
| `3_explain.py` | One label and one rule per unit, written by Claude Opus 5.5, blind to why the unit was picked |
| `4_analyze.py` | Applies the pre-registered rules; writes `results/verdicts.json` |
| `make_figures.py` | Draws every figure and prints the numbers it plots |
| `reproduce.sh` | Every command in order (`--dry-run` prints them) |
| `results/` | Everything the note reads (`results/inputs/` holds the Pigment Gym rollouts the explainer and the KL set use) |

The checkpoints come from Pigment Gym's Qwen3 runs (`experiments/pigment-gym/`), read from a Modal volume.
