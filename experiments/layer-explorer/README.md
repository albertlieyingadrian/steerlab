# Layer explorer: watch a model pick its next word

**A guide to `steerlab inspect`**: type a few words and watch Qwen2.5-0.5B-Instruct arrive at its next word, layer by
layer. Which layers push it, which heads and neurons, what a neuron writes, what happens when you switch it off, and
whether it matters more than 20 random neurons from the same layer. Inspired by
[neuronaut](https://github.com/andrew-morris/neuronaut) and its [web game](https://neuronaut.ishaanko.com).

**Read the guide:** [report](report/index.html) · **try it without installing:** [the saved explorer](report/explorer/)

```bash
uv run steerlab inspect            # the live app at http://127.0.0.1:8770 (add --download the first time)
```

## The sanity checks

| Check | Result |
|---|---|
| Embed + heads + neurons = the real target logit (final norm scale held fixed) | largest leftover 1.2e-5 on 8 prompts |
| Logit lens after the last layer = the real output | same top word 8/8, largest difference 2.5e-5 |
| Unedited run = plain model | bit-identical scores 8/8 (and with a ×1 edit), identical greedy continuations 8/8 |
| Saved dose tests against 20 random neurons of the same layer | 19/23 move the target more than twice as far as the most effective random neuron |
| L21.N4090's label ("法国", " French", " France") on 10 + 10 prompts | peak > 5 on 9/10 France prompts, 0/10 others |

## Files

| File | What it does |
|---|---|
| `sanity.py` | the five checks; writes `results/sanity.json` |
| `make_cover.py` | the cover and thumbnail: screenshots of the saved explorer on the France prompt with layer 21 open |
| `screenshots.py` | still screenshots of the saved explorer (Figure 8) |
| `record_demos.py` | the 20-second overview and Figures 1–7: short looping videos recorded from the live app (`report/media/`) |
| `reproduce.sh` | all of the above, plus the export and the package tests (`--dry-run` prints the commands) |
| `report/explorer/` | the saved explorer: the app's page reading `data.json` (six presets), written by `steerlab inspect --export` |

The code is in `packages/steerlab/src/steerlab/inspect/` (`core.py` the numbers, `app.py` the routes, `export.py`
the presets and the static export, `static/` the page) and `packages/steerlab/src/steerlab/cli_inspect.py`.
