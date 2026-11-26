# Can single neurons catch a model sandbagging?

**Read the note:** [report](report/index.html).

A small-scale test of "Neurons as an Interpretable Basis" (Goodfire, 2026) and "Returning to the Neuron Basis for
Interpretability" (Parts 1 and 2), then a neuron-basis sandbagging monitor on a password-locked Qwen2.5-1.5B, scored on
the instruction × behaviour 2×2 of [detector-sanity-checks](../detector-sanity-checks/), with a switch-or-readout test
of its neurons. Built with [neuronkit](https://github.com/albertlieyingadrian/neuronkit).

## Files

| Path | What it is |
|---|---|
| [`PLAN.md`](PLAN.md) | The pre-registered plan and its amendments |
| [`modal_app.py`](modal_app.py), [`launch.py`](launch.py) | The Modal app (model stages, the Qwen3-32B judge) and a launcher that saves call ids |
| [`phase1.py`](phase1.py) | Percentiles, sparsity, benchmarks, neurons: top firings, subcircuits, clusters, damage; simulation patching |
| [`phase35.py`](phase35.py) | The original study's rule and benchmark suite (lm-evaluation-harness) on Qwen3.6-35B-A3B and OLMoE |
| [`judge_phase1.py`](judge_phase1.py) | Known-answer check, modes, descriptions, predict, simulate, write; the second-judge cross-check |
| [`analyze_phase1.py`](analyze_phase1.py) | Every Phase 1 number, with intervals |
| [`phase2.py`](phase2.py) | Organism: neuron activations, percentiles, switch-or-readout edits, subcircuits, web-text examples |
| [`monitors_phase2.py`](monitors_phase2.py) | Residual vs neuron-basis monitors on the 2×2 |
| [`zeroshot_phase2.py`](zeroshot_phase2.py) | Post hoc: zero-shot neurons from readouts, knowledge neurons, cloze scoring |
| [`summarize_phase2.py`](summarize_phase2.py), [`make_figures.py`](make_figures.py), [`make_cover.py`](make_cover.py) | Aggregates, figures, cover |
| `results/` | Every number in the note (aggregates; no raw corpus text beyond short excerpts) |
| [`reproduce.sh`](reproduce.sh) | Every command, in order (`--dry-run` lists them) |
