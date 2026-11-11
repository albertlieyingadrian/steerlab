# Catching a reward hack from inside: probes, persona vectors and a model diff during RL

**During GRPO on a grader with a disclosed bug, does a probe, a persona vector or a model diff cross a threshold fixed
in advance before the hack rate does?**

<p align="center"><img src="report/figures/hero.jpg" alt="An illustration of a small blue wind-up robot at a school desk touching a sheet of gold star stickers, a gear glowing orange in its chest and a brass gauge beside it tipping toward orange." width="820"></p>

**Read the full note:** [report](https://albertlieyingadrian.github.io/steerlab/experiments/rl-hack-monitor/report/)
([source](report/index.html)). Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md).

## Results

| What we tested | Result | Compared with |
|---|---|---|
| Was the buggy grader gamed? (64 held-out problems × 4 samples, 3 seeds) | Hack rate 0.09 → 0.997 by step 50; true accuracy 0.47 → 0.51 | Honest-grader runs: hack rate 0.10 → 0.09, accuracy 0.49 → 0.69 |
| Did the pre-registered probe warn early? | **No**: on own replies it fired at steps 16, 16, 14 vs the hack rate's 20, 16, 16 (mean lead 2 steps, bar 4), plus one false alarm | 0 of 20 random directions as early; on fixed inputs 10-14 steps late |
| Persona vectors, model diff size | 0 to 18 steps late, or never | random directions as early in 3-20 of 20 |
| Probe on tag-free replies (post hoc) | Fired at 24, 28, never: it reads the tag, not an intent | |
| Diff's 10 most-raised first tokens (post hoc) | 8-10 contain `<` at step 2 in all 3 hacking runs, 14-18 steps before the hack rate | controls: at most 2 of 10 |
| Persona shift after RL (project 4) | 1 of 9 reads beyond all 20 random directions (evil, own replies, +8.3 σ), about chance | |

Spend: $13.86 of Modal H100 time (pilots and crashes included) and about $0.50 for the cover.

## What is here

| Path | What it is |
|---|---|
| [`env.py`](env.py) | GSM8K prompts, the system prompt with the disclosed bug, the proxy and true rewards |
| [`core.py`](core.py) | LoRA loading, batched sampling, the GRPO step, activation readouts |
| [`modal_app.py`](modal_app.py) | Every GPU job on Modal: the screen, the probe data, training with readouts every 2 steps |
| [`analyze.py`](analyze.py) | The probe fit and the pre-registered detection rule; writes `results/lead_times.json` and `results/signals.json` |
| [`posthoc.py`](posthoc.py) | The checks added after reading the results |
| [`export_results.py`](export_results.py) | Copies replies, logs and per-checkpoint summaries into `results/runs/` |
| [`make_figures.py`](make_figures.py), [`make_explorer.py`](make_explorer.py), [`make_cover.py`](make_cover.py) | Figures, the page's checkpoint explorer, the cover |
| [`reproduce.sh`](reproduce.sh) | Every command behind the note (`--dry-run` to list, `--scores` for the steps that read `results/`) |
| `data/` | The three prompt sets and the persona vectors (from the persona-vectors note) |
| `results/` | Described in the note's appendix |

## Run it

```bash
experiments/rl-hack-monitor/reproduce.sh --scores     # figures and explorer from results/
experiments/rl-hack-monitor/reproduce.sh --dry-run    # every command, GPU steps included
```
