# sandbagkit

The probing core that steerlab is built on: read and edit a language model's hidden activations, build
difference-of-means steering vectors, and score linear probes honestly. Everything comes back as typed results with
confidence intervals.

- `sandbagkit.probing`: forward hooks that read or replace a layer's output (and are always removed); activation
  capture by teacher forcing, averaged over the reply's tokens; steering vectors with a guard against train/test
  leakage; linear probes with cross-validation, bootstrap intervals, AUROC and a shuffled-label baseline.
- `sandbagkit.core`: typed trials and results, statistics (bootstrap, Wilson, chi-square, AUROC, permutation tests).
- `sandbagkit.backends`: a local Hugging Face backend that picks the device and a safe dtype, and a tiny random
  model that makes every test run offline in milliseconds.

The name comes from where it started: rebuilding an activation probe that detects AI *sandbagging* (a model
deliberately underperforming). steerlab grew out of its steering-vector code.

## Install

```bash
pip install "sandbagkit[probing]"    # torch, transformers, scikit-learn, peft; the base install never imports torch
```

In this repo, `uv sync --all-packages` installs it with steerlab, editable.

## Use

```python
from sandbagkit.probing import extract_activations, steering_vector

acts = extract_activations(model, tokenizer, prompts, [14], responses=replies, chat_template=True)[14]
vector = steering_vector(acts, labels, groups=pair_ids, layer=14)  # mean(label 1) - mean(label 0)
```

MIT licensed.
