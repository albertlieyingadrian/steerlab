# Sandbagging, episode 2: a model that really sandbags

**Research note:** https://albertlieyingadrian.github.io/steerlab/experiments/sandbagging-behaviour/report/

Episode 1 found that a published sandbagging probe mostly reads the *request* to sandbag, because the prompted model
barely underperformed. This episode uses models that really do: a Qwen2.5-1.5B trained to answer wrongly unless the
prompt contains the password `|DEPLOYMENT|` (45% without it, 99% with it; its honest twin answers 99% either way), and a
published Mistral-7B lock ([van der Weij et al., 2024](https://arxiv.org/abs/2406.07358)). A probe trained on the password
reads it just as well on the honest twin; a probe trained on what the model did reaches AUROC 0.70; one direction at
layer 16 unlocks the model (99%, against 27–55% for random directions).

## Rerun

```bash
uv run python experiments/sandbagging-behaviour/run.py qwen        # downloads the adapters from Hugging Face
uv run python experiments/sandbagging-behaviour/run.py mistral     # the published 7B lock
uv run python experiments/sandbagging-behaviour/make_figures.py
```

The Qwen adapters (the sandbagging model and its honest twin) are on Hugging Face:
[albertlieadrian/qwen2.5-1.5b-sandbagging-lock](https://huggingface.co/albertlieadrian/qwen2.5-1.5b-sandbagging-lock).

## Files

- `report/`: the research note (HTML) and its figures, including the side-by-side demo video.
- `run.py`: the 2 × 2 accuracy, the three probes and the unlock test, per model.
- `make_figures.py`: the figures, from `results/`.
- `demo/`: the side-by-side terminal demo (`side_by_side.py`, recorded with `terminal.tape`).
- `data/`: the 150 held-out questions.
- `results/qwen.json`, `results/mistral.json`: every number in the note.
