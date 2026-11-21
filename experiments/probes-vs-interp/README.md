# Small probes vs interpretability (The Taste Stack, Episode 1)

When you want to detect AI-written text and know *why* a detector or a reward model says what it says, should you use
a small probe or read the model's features? Ten tests, nine pre-registered predictions.

Note: `report/index.html` · dev ledger: `results.tsv`.

## Files

| File | What it does |
|---|---|
| `common.py` | paths, text hygiene, bootstrap AUROC and means, read-only access to Sounds Like Me's data |
| `budget.py` | Modal hard cap ($200) in code; every paid call goes through `budget.paid` and lands in `results/spend.jsonl` |
| `build_data.py` | M0: human sets, topic- and length-matched AI text, humanizers, shortcut edits, confound and hack sets |
| `modal_app.py` | Modal: vLLM generation and judge, DIPPER, activations + SAE features + neurons, top windows, steering |
| `probes.py` | tests 1, 2, 3, 6: every detector on one leaderboard, learning curves, cost per 1k texts |
| `causal.py` | test 4: steer or ablate what the detectors read vs norm-matched random directions |
| `explain.py` | tests 5 and 7: SAE / neuron / bag-of-words / style / example readouts, blind reader, grader |
| `modal_pigment.py`, `pigment.py` | test 8: internal signals vs the outside check over Pigment Gym's saved checkpoints |
| `modal_voice.py`, `voice.py` | tests 9 and 10: per-author voice vector vs LoRA vs few-shot prompting |
| `analyze.py` | scores P1-P9 -> `results/predictions.json` |
| `make_figures.py` | every figure in the note, from `results/` |
| `reproduce.sh` | the whole pipeline (`--dry-run` prints the commands) |

## Data rules

- Human text: Enron sent mail and the Blog Authorship Corpus (public), via Sounds Like Me's splits, read only.
- AI text: open models (Qwen3-8B, Llama-3.1-8B-Instruct, gemma-3-27b-it, phi-4, Qwen3-32B) writing from each human
  text's corpus summary; Sounds Like Me's drafts. No customer text anywhere.
- `data/` is gitignored and rebuilt by the scripts; `results/` holds every number the note quotes, and
  `results/data_manifest.json` the hashes of the frozen sets.
