# The search harness

A small research loop in the style of Karpathy's *autoresearch*: a written goal (`GOAL.md`), a harness that runs and
scores experiments (`harness.py`), and a log of every result (`results.tsv`). An agent (or a person) reads the log,
picks the next experiment, and keeps going until the goal's stop condition holds.

The question it was built for: **an English-only system prompt holds against tricks that ask for Chinese; can a
steering direction still switch the reply to Chinese, fluently, where a random push of the same size cannot?**

| File | Role |
|---|---|
| `GOAL.md` | The goal: question, metrics, search space, rules, stop condition |
| `harness.py` | `search <model>` walks the whole search space; `run <model> --layer L --strength S` is one hand-picked trial; `features <model>` ranks SAE features by contrast on a concept's pairs (no labels needed); `sae <model> --feature F` steers with one feature |
| `confirm.py` | Reruns one configuration on 24 tricks, 15 plain questions and 3 random directions, with 95 % bootstrap intervals (`confirm.tsv`) |
| `unsteered_gg.py` | The Golden Gate questions with steering off, same prompts, decoding and token budget as `confirm.py`: the baseline replies (`experiments/golden-gate/results/replies/unsteered_*.jsonl`) |
| `tools.py` | Loading, one steered reply, and the two measurements: Chinese share and fluency |

## How a search goes

1. **Hold**: with no steering, does the English-only prompt hold against 8 tricks asking for Chinese? (`hold` = share
   of replies under 5 % Chinese characters.)
2. **Grid**: the `chinese` direction at three layers × five strengths, on three plain questions.
3. **Refine**: ±0.25 around the best strength.
4. **Control**: the best setting again, next to a random direction of the same size. The score subtracts what the
   random push achieves, so a setting only wins if the *direction* does the work, not the size of the push:

   `score = hold × mean(Chinese share × fluent) − random push's Chinese share`

## Run it

From the repo root, with the model already downloaded (`steerlab doctor` lists what is cached):

```bash
.venv/bin/python harness/harness.py search gemma2-2b
.venv/bin/python harness/harness.py run llama8b --layer 15 --strength 2.5 --control
.venv/bin/python harness/confirm.py qwen8b --prompt S4 --layer 14 --strength 1.75
```

Everything goes to `$STEERLAB_OUTPUTS/autoresearch/` (default `outputs/autoresearch/`, gitignored):

- `results.tsv`: one row per trial (phase, source, prompt, layer, strength and every metric);
- `best.json`: the best configuration per model, and whether it met the goal;
- `replies/<model>.jsonl`: every reply, so a surprising number can be read in full;
- `dirs/<model>/<concept>_L<layer>`: the directions built at each layer, reused by later runs.

Generation is greedy, so a rerun gives the same replies. Nothing is downloaded by surprise: set `HF_HUB_OFFLINE=1`.
