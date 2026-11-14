# circuitkit

Three tools on top of [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) attribution graphs:

| Tool | Question it answers | Runs without a GPU on |
|---|---|---|
| `circuitkit.diff` | Which features and edges did fine-tuning or RL add or remove, and is the change real? | saved graph summaries |
| `circuitkit.validate` | Is a group of graph features a **switch** (it causes the behaviour) or a **readout** (it only reflects it)? | the built-in toy model |
| `circuitkit.labels` | What does each feature do, in a few words, and how far can that label be trusted? | cached model replies |

It grew out of steerlab's circuits episodes. In
[a password lock](https://albertlieyingadrian.github.io/steerlab/experiments/lock-circuit/report/) and
[an RL reward hack](https://albertlieyingadrian.github.io/steerlab/experiments/rl-hack-circuit/report/), the graph's
most influential features all turned out to be readouts. This package turns the checks behind that finding into
reusable code.

## Install

```bash
pip install "steerlab-circuitkit @ git+https://github.com/albertlieyingadrian/steerlab#subdirectory=packages/circuitkit"
# real models: circuit-tracer from GitHub, plus torch
pip install "git+https://github.com/decoderesearch/circuit-tracer"
# labels: pip install "steerlab-circuitkit[anthropic]" (Claude Haiku 5.5, the default) or "[openai]"
```

## Five-minute tour (no GPU, no key)

```bash
circuitkit validate --demo --html validate.html
circuitkit diff examples/data/rl_hack_graphs.json.gz --before installed/before --after installed/after \
  --control-before control/before --control-after control/after --html diff.html
```

```python
from circuitkit.validate import logit_diff, validate
from circuitkit.validate.toy import demo, demo_nodes

backend = demo()
metric = logit_diff(backend.token_id(" Austin"), backend.token_id(" Sacramento"))
report = validate(backend, "dallas", demo_nodes(backend), metric=metric, counterfactual="oakland")
print([(g.name, g.verdict.value) for g in report.groups])
# [('top-5', 'switch'), ('L5/100@5', 'switch'), ('L7/200@5', 'readout'), ('L8/201@5', 'readout'), ...]
```

On a real model, `circuitkit.tracer.TracerBackend` wraps a circuit-tracer `ReplacementModel`:

```python
from circuit_tracer import ReplacementModel
from circuitkit.tracer import TracerBackend

model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")
report = validate(TracerBackend(model), prompt, nodes, metric=metric, counterfactual=other_prompt)
```

## What each tool checks

**diff.** Graphs of the same prompts before and after training are reduced to per-feature influence and
feature-to-feature edges, summed over positions. Each feature gets a status (gained, lost, up, down, stable). The
statuses are backed by:

- a paired swap null: before and after labels are swapped at random within each prompt pair, with a
  Benjamini-Hochberg correction across features;
- a bootstrap interval and a split-half check;
- optionally, a control run (did the change happen only in the run that learned the behaviour?) and a replication
  seed.

The validity check compares replacement and completeness scores, error-node influence and per-layer reconstruction
error (`circuitkit.tracer.reconstruction_error`) before and after. The transcoders were trained on the base model,
and the report says so every time.

**validate.** For each group of features:

- ablation (set them to 0);
- steering (multiply them by 3);
- a counterfactual patch (copy their values from a prompt with a different answer);
- each of the above on 20 random feature sets matched by layer and position, and on 20 sets matched by write size
  (|activation| x decoder norm).

The verdict follows printed rules: switch, partial, backup, nonspecific or readout. An optional explanation swap (Gao's
test) runs label rules written as code in a checked sandbox. Output is JSON plus an HTML page.

**labels.** The pipeline:

1. Explain each feature from its top activating records.
2. Score the label on held-out records: detection AUROC (activating and silent excerpts) and, optionally, simulation
   correlation.
3. Report calibration: stated confidence against the earned score, and the detector's expected calibration error.

Every reply is cached. Labels export in Neuronpedia's explanation format, and Neuronpedia's own labels can be scored by
the same detector for a fair comparison.

## Results that reproduce the episodes

- **Graph diff.** On the RL-hack study's 48 graphs, `diff_graphs` picks the same ten features the study did, in the
  same order. 9 of the 10 are specific to the run that hacked and 6 come back in another seed (test:
  `tests/test_diff.py::test_reproduces_rl_hack_selection`). Split halves of the prompts agree on 70% of the top 10, and
  reconstruction error is unchanged by RL (0.500 vs 0.499).
- **Labels.** Claude Haiku 5.5 labelled 50 Gemma-2-2B transcoder features for $0.26 in total. Its labels match
  Neuronpedia's on detection (AUROC 0.80 vs 0.78 and 0.80) and beat them on simulation (r = 0.45 vs 0.37 and 0.38).
  The explainer's stated confidence barely predicts its score (r = 0.11), so use the score, not the confidence.
- **Live validator runs** on the RL-hack model and the password-locked model: see `examples/data/live/`.

## Layout

```
src/circuitkit/
  diff/       graphs.py (summaries and loaders: circuit-tracer .pt, Neuronpedia JSON), core.py (the diff)
  validate/   core.py (checks and verdicts), controls.py, metrics.py, sandbox.py, toy.py
  labels/     autointerp.py (explain, detect, simulate, calibrate), providers.py (Claude, OpenAI, cache), neuronpedia.py
  tracer.py   circuit-tracer adapter and reconstruction error
  report.py   HTML pages
  cli.py      the circuitkit command
examples/     notebooks 01-03, data/, scripts/ (how the bundled data was made), modal/ (live GPU checks)
```

## Development

```bash
uv pip install -e "packages/circuitkit" pytest ruff mypy nbclient ipykernel
cd packages/circuitkit && pytest && ruff check && mypy src
```

MIT licence.
