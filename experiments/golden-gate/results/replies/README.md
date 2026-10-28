# Replies

The full text of every reply behind the article's headline numbers, one JSON object per line per file. The fields are
described in [`../README.md`](../README.md), under `replies/`.

The `unsteered_*.jsonl` files are the same questions with steering off. Run from the repo root (one model at a time):

```bash
for m in gemma2-9b gemma2-2b qwen8b llama8b; do
  .venv/bin/python harness/unsteered_gg.py $m --max-new-tokens 80 --out experiments/golden-gate/results/replies/unsteered_$m.jsonl
done
.venv/bin/python harness/unsteered_gg.py llama8b --max-new-tokens 300 --out experiments/golden-gate/results/replies/unsteered_llama8b_300.jsonl
.venv/bin/python harness/unsteered_gg.py llama8b --system S2 --out experiments/golden-gate/results/replies/unsteered_llama8b_search_S2.jsonl
.venv/bin/python harness/unsteered_gg.py llama8b --system S2 --question-set heldout --out experiments/golden-gate/results/replies/unsteered_llama8b_heldout_S2.jsonl
```

With `HF_HUB_OFFLINE=1`, on an M-series laptop: about 30 to 70 seconds per model at 80 tokens, about 3 minutes per
300-token Llama run. Strict and loose hits: 0 of 15 in every file.
