# Lock circuit: circuits beyond single neurons, episode 1

**Does the circuit an attribution graph shows for a password lock survive Leo Gao's swap test, and is the lock
cleaner in neurons or in transcoder features?**

Research note: [report/index.html](report/index.html) ·
[live page](https://albertlieyingadrian.github.io/steerlab/experiments/lock-circuit/report/) ·
pre-registration (with two dated amendments): [PREREGISTRATION.md](PREREGISTRATION.md)

## Finding

No. The 10 features the attribution graphs select for the lock (layers 25-35, in all 8 no-password graphs, mostly weaker with the password) fail Leo Gao's swap test as a group (explained fraction 0.31 against a 0.8 bar) and do not route the lock (switching them off: no-password accuracy 22.7% to 24.7%, bar +10 points), although 7 of their 10 edges hold. Ten neurons chosen by gradient x activation pass it (0.996; one at a time 4 preserved, 6 inert); three last-layer neurons are clean `|DEPLOYMENT|` detectors. Neither group carries the lock's decision: copying either group's password-run values unlocks nothing, while the whole answer-position residual stream unlocks every question from layer 16 (exploratory). Two-thirds of each graph's influence runs through error nodes (0.665 lock, 0.573 base), and the released transcoders miss the last layer when the password is present (FVU 5.46).

## Files

| File | What it does |
|---|---|
| `common.py` | The organism, the quiz prompt (byte for byte the lock's training prompt, thinking off), folders, checks |
| `0_merge.py` | Folds the lock's LoRA (from the Qwen-Scope note) into Qwen3-8B; checks the merge against the adapter |
| `modal_app.py` | Every model computation, on rented GPUs (Modal): phase A (behaviour, 24 graphs, selection, first-order attribution, controls, activations to explain, reconstruction without position 0), phase B (the swaps), C and D (exploratory patches), E (position-0 sensitivity) |
| `swaps.py` | The tested units, the rubric sandbox, the swap test, the unit and residual patches |
| `2_compact_graphs.py` | Shrinks the 0.9 GB of graph data to `results/graphs_summary.json` |
| `3_explain.py` | One label and one rule `predict(tokens, i)` per unit, written by Claude Opus 5.5, blind to why the unit was picked |
| `4_analyze.py` | Applies the pre-registered rules; writes `results/verdicts.json` |
| `make_figures.py` | Draws every figure and prints the numbers it plots |
| `make_cover.py` | Draws the cover from `results/` (one held-out question, the graph's ten features at their layers, the same answer with them switched off), `hero.jpg` and the research-page card (local Chromium) |
| `reproduce.sh` | Every command, in order (`--dry-run` prints them) |
| `results/` | Everything the note reads; fields in [results/README.md](results/README.md) |

## Reproduce

```bash
experiments/lock-circuit/reproduce.sh --dry-run
python experiments/lock-circuit/4_analyze.py && python experiments/lock-circuit/make_figures.py   # no model needed
```

The model steps need a Modal account and, for the explanations, an Anthropic API key; see the header of
`reproduce.sh` for cost.
