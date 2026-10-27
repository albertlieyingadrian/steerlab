# A sandbagging probe that reads the request, not the behaviour

**Research note:** https://albertlieyingadrian.github.io/steerlab/experiments/sandbagging-probe/report/

A published activation probe for AI sandbagging ([Detecting AI Sandbagging](https://subhadipmitra.com/blog/2025/detecting-ai-sandbagging/),
with the [ai-metacognition-toolkit](https://github.com/bassrehab/ai-metacognition-toolkit)'s 32 contrast pairs), rebuilt with
[sandbagkit](../../packages/sandbagkit/) on Qwen2.5. It replicates (92% CV on Qwen2.5-7B), but a one-line keyword rule
scores AUROC 0.89 against its 0.99 and the model barely sandbags. A direction that does cause sandbagging (accuracy
83% → 12% when added) still tracks the instruction more than the wrong answers.

## Rerun the core comparison (about 20 seconds)

```bash
uv run python experiments/sandbagging-probe/probe.py                                   # Qwen2.5-0.5B
uv run python experiments/sandbagging-probe/probe.py --model Qwen/Qwen2.5-7B-Instruct --layers 12
```

## Files

- `report/`: the research note (HTML) and its figures.
- `probe.py`: the probe with its interval and shuffled baseline, next to the keyword rule.
- `results/findings.tsv`: every number in the note, with its model and detail.
- `results/probe_rerun.json`: `probe.py`'s output on Qwen2.5-0.5B.
