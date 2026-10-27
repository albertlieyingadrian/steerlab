# Results: sandbagging, episode 2

Written by [`run.py`](../run.py), one file per model; the [research note](../report/) and
[`make_figures.py`](../make_figures.py) read them.

- [`qwen.json`](qwen.json): the Qwen2.5-1.5B sandbagging model and its honest twin.
- [`mistral.json`](mistral.json): the published Mistral-7B lock.

Each file holds the 2 × 2 accuracy (model × password), the three probes by layer (password, model, behaviour) with
their intervals, and the unlock test against random directions of the same length.
