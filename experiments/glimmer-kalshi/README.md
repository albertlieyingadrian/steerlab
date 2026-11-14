# glimmer-kalshi: replicating Fireworks' Muse Glimmer decision tuning on Kalshi

A replication of Fireworks' cookbook `training/examples/decision/kalshi_v1` ("Train Muse Glimmer for free"): Muse
Glimmer 30B trained with LoRA and a K-way cross-entropy on option letters, over 20,051 questions from settled Kalshi
markets, then calibrated with one temperature per question type. Plus an exploratory twist on Qwen3-8B: is the
model's confidence readable from its activations, and what does a temperature fix? The note is
[`report/index.html`](report/index.html); the plan, fixed before training, is [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Pipeline

| Step | Command | Where it runs | Output |
|---|---|---|---|
| Data | `huggingface_hub` download of `fireworks-ai/kalshi-decision-v1` (checked against `data/manifest.json`) | laptop | `data/{train,dev,test,unseen_family}.jsonl` |
| Train + readouts | `fw_train.py --output-model-id kalshi-glimmer-replica` (cookbook on `PYTHONPATH`) | Fireworks serverless training (free) | `readouts/{base,tuned}_{dev,test}.jsonl` |
| Score | `evaluate.py` | laptop (numpy) | `results/metrics.json`, `results/test_probs.jsonl` |
| Kalshi markets | `kalshi_events.py`, then `backtest.py fill`, `backtest.py map` | Kalshi public API | `kalshi/map.jsonl`, `kalshi/map_stats.json` |
| Quotes | `backtest.py quotes` | Kalshi public API (about 4 requests/s) | `kalshi/quotes.jsonl` |
| Backtest | `backtest.py run` | laptop | `results/backtest.json` |
| Twist: Qwen3-8B | `modal run modal_qwen.py --phase all`, then `--phase probe` | Modal H100, then Modal CPU | `qwen/*.jsonl`, `results/qwen_probe*.{json,npz}` |
| Twist: scoring | `evaluate.py --dir qwen --out qwen_`, `meta_baseline.py`, `qwen_stats.py` | laptop | `results/qwen_*.json` |
| Figures | `make_figures.py` | laptop | `report/figures/*.png` |

`reproduce.sh --dry-run` prints every command in order.

## Results files

- `readouts/{base,tuned}_{dev,test}.jsonl`: one row per labeled question, read on the Fireworks trainer.
  `row` (line in the split file), `qid`, `type` (`noul` yes/no or `choice`), `n_tokens` (prompt length),
  `gold` (index of the right option; yes/no questions are A = yes, B = no), `logprobs` (the K option-letter
  log-probabilities at the answer position, before any temperature).
- `results/metrics.json`: per model (`base`, `tuned`) and setting (`raw` T=1, `calibrated` dev-fitted T per type):
  accuracy with a 95% event-bootstrap interval, Brier sum, ECE (15 bins), mean confidence, the confidence curve,
  the reliability bins, the same on the unseen-topic slice (the last 765 test rows) and by question type; the
  common-answer reference; the paired tuned-minus-base difference.
- `results/test_probs.jsonl`: per test question, the option probabilities under each model and setting.
- `kalshi/map.jsonl`: one row per Kalshi contract behind a test question: `row`, `qid`, `opt` (option index;
  0 for yes/no), `won` (from the dataset label), `ticker`, `series_ticker`, `open_time`, `close_time`, `result`
  (Kalshi's settlement), `hist` (whether Kalshi serves it from its historical archive).
- `kalshi/quotes.jsonl`: per ticker, the yes bid and ask (dollars) of the last completed hourly candle at or
  before open + 2 h, and that candle's end time.
- `results/backtest.json`: per model and subset, events and bets, ending value of $1,000 and its 90% interval.
- `qwen/*.jsonl`: the Qwen3-8B readouts, same columns as `readouts/`.
- `results/qwen_probe.json`, `results/qwen_probe_preds.npz`, `results/qwen_meta_baseline.json`,
  `results/qwen_stats.json`: probe AUROC/ECE per layer, the per-question probe outputs, the metadata-only baseline
  and the event-bootstrap intervals.

## Cost

Fireworks: $0 (serverless training on Muse Glimmer is free until 2026-11-30; evaluation reads logprobs on the same
trainer, so no inference endpoint or deployment was used). Modal: $3.04 for the Qwen3-8B twist (H100 readouts and training, CPU probes). Kalshi's public API needs no key.
