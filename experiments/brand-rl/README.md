# An RL environment for "on brand, not slop, not a copy"

Code and data for the steerlab note `report/index.html` (The Taste Stack, Episode 6). A small model (Qwen3-4B
Instruct, LoRA) writes one HTML page per brief for a public service that uses the GOV.UK Design System. Each page is
rendered in a browser and scored by the brand checker from Episode 5 (per rule, with the source line), a coverage
check against the brief, the markup slop probe from Episode 3 and the copy gate from Episode 5. Four GRPO arms
(the checker alone, the full environment, the environment plus a novelty term, and a shuffled-reward control) ask
whether RL learns the brand or games the checker, whether a novelty term slows collapse to the mean, and whether an
internal monitor built from the base model catches hacks early, against norm-matched random directions.

Run everything with `reproduce.sh` (`--dry-run` prints the commands). Training and sampling need a Modal account
(one H100 per run); the rulers, analysis and figures run on a laptop with Playwright's Chromium.

## Files

| File | What it does |
|---|---|
| `ep6.py` | Paths, the policy, the 16 rule families and their split, the briefs (train 600, dev 48, held-out 96, monitor 96) and the prompt |
| `render.py` | Headless Chromium at 1280 × 800, scripts and network off; Episode 5's and Episode 2's extractors in one load |
| `reward.py` | Every score a page gets: rule families from Episode 5's proof artifact, coverage, fidelity, slop probe, copy gate, novelty; each arm's reward |
| `slop_fit.py` | Freezes Episode 3's markup and screenshot probes as weights (`results/slop_probes.json`) and checks the refit |
| `ruler.py` | The pass bars (renderer, reward mapping, brief rulers on oracle pages, copy gate, loop flag): `results/ruler.json` |
| `hacks.py` | The monitor's constructed hacks (drop, div soup, copy, cut) of the base model's honest pages; fits the direction and probe (`results/monitor.npz`) |
| `modal_rl.py` | The Modal app: GRPO training with checkpoints and resume, sampling with scores and activations, the steering test, the screenshot probe |
| `launch.py`, `drive.py`, `status.py` | Spawn runs after the budget check; the detached driver (respawn on failure, then held-out evals); a log reader |
| `analyze.py`, `make_figures.py` | Every number with its bootstrap interval (`results/summary.json`) and the figures |
| `budget.py` | The Modal cap in code ($450 soft, $500 hard), from Modal's billing API |
| `configs/` | The family split, the novelty scale, the pilot and main run configs |
| `results.tsv` | Every dev tuning try (autoresearch ledger) |

## What is not in the repo

Policy pages, screenshots and activations stay on the Modal Volume `taste-ep6` and in `~/.cache/steerlab/brand-rl/`.
The note ships aggregate numbers, figures and a small set of example pages chosen for the explorer.
