# circuitcheck: the circuits episodes' checks as a toolkit (circuits beyond single neurons, episode 3)

**Can the checks behind episodes 1 and 2 (controls, nulls, swap tests, validity) be packaged so that anyone with
circuit-tracer can run them, and do they reproduce the episodes' verdicts?**

Research note: [report/index.html](report/index.html) ·
[live page](https://albertlieyingadrian.github.io/steerlab/experiments/circuitkit/report/) ·
package: [circuitcheck](https://github.com/albertlieyingadrian/circuitcheck) (its own repo; renamed from circuitkit) · docs: [`docs/`](https://github.com/albertlieyingadrian/circuitcheck/tree/main/docs) ·
episodes: [lock-circuit](../lock-circuit/), [rl-hack-circuit](../rl-hack-circuit/)

## Files

| File | What it does |
|---|---|
| `make_figures.py` | Recomputes every number in the note from the package's shipped data, writes `results/summary.json` and draws the figure and cover |
| `reproduce.sh` | Every command behind the numbers, in order (`--dry-run` prints them) |
| `results/summary.json` | The note's numbers: the diff on the RL-hack graphs, the live validator runs, the label scores |

The data lives with the package, in [circuitcheck's `examples/data/`](https://github.com/albertlieyingadrian/circuitcheck/tree/main/examples/data):
- `rl_hack_graphs.json.gz`: 48 graph summaries from episode 2;
- `live/rl_hack.json`, `live/lock.json`: the live validator and reconstruction runs on Modal;
- `labels_gemma_*`: 50 Gemma-2-2B features, their labels and scores, and every cached model reply.
