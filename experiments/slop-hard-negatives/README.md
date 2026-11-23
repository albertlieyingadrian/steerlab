# slop-hard-negatives: can a slop detector fix its own mistakes?

Research note: [report/index.html](report/index.html), published at
<https://albertlieyingadrian.github.io/steerlab/experiments/slop-hard-negatives/report/>.

The slop-detector note's Qwen3-1.7B detector (4-bucket head, data v2b) flags some old human web pages as AI. This
experiment scores 250,000 FineWeb pages crawled before November 2022, adds the pages it scores highest back as human
training data, retrains with the same recipe and repeats: hard-negative rounds. Two controls: the same pages plus an AI
mirror of each (does it learn "old topics are human"?) and the same number of random old pages (is it the *hard* pages
that help?). Every round is scored on held-out old web pages (other sites), magazine articles, Enron emails, EditLens,
Epoch AI's imitations and fresh pages from four 2026 open models, and read inside with probes and ablations.

The label rule (crawled before November 2022 = human) is fixed and external, so this is self-training, not recursive
self-improvement.

## Scripts, in order

| Script | What it does |
|---|---|
| `hn.py` | Paths, the fixed design numbers (cut, K, rounds, weights), the spend ledger with its hard cap, metrics |
| `modal_hn.py` | Modal app on the modal4 workspace: `build_pools` (CPU), `train_round`, `evaluate`, `score_pool` (H100) |
| `pools.py` | FineWeb sampling (8 dumps, 2017-04 to 2022-40, plus a 2025 dump for topic seeds), split by host; checks dates |
| `gen.py` | Open-weight generation and judging through OpenRouter: AI mirrors, the 2026 set, the judge and its known-answer check |
| `loop.py` | `prep` (freeze evaluation and interp sets), `pilot`, `round0`, `arm a|b|c`, `seeds` |
| `evaluate.py` | Every metric with paired bootstrap intervals (`results/metrics.json`); `preds` scores the plan's predictions |
| `interp.py` | Probes and mean-ablations on the head's input per round (`results/interp.json`) |
| `noise.py` | Judged machine-made share of mined pages; the blind reference sample and its check |
| `make_figures.py` | The note's figures, printing the numbers each one plots |
| `reproduce.sh` | Every command; `--scores` recomputes from saved scores, `--dry-run` lists them |

## Results files

| File | Contents |
|---|---|
| `results/metrics.json` | Per run: FPR and recall at the fixed cut with 95% intervals, AUROC, each run's own refitted cut, paired differences, drift, mined-set stats |
| `results/scores/<run>.json` | One score per evaluation item, by set, in the order of `results/evalset.json` |
| `results/evalset.json` | Each evaluation set's labels, kinds and item ids (no text) |
| `results/mined/<arm>_r<k>.json` | Each round's chosen pages: pool index, id, score, how many sat above the cut |
| `results/interp.json`, `results/interpset.json` | Probe AUROC, reliance (mean-ablation) and random bands per run; the concept sets' labels |
| `results/noise.json`, `results/judge_known*.json`, `results/ref_*_check.json` | Judge results, its known-answer checks and the blind reference checks |
| `results/train/<run>.json`, `results/val/<run>.json` | Training summaries; validation scores (for each run's own cut) |
| `results/predictions.json` | The plan's predictions, scored |
| `results/spend.jsonl` | Every paid call: reservation, then measured cost |
| `results/pools.json` | Pool sizes, dates, dumps, host overlap |

No page text is committed: FineWeb pages keep their own rights, and EditLens is CC BY-NC-SA 4.0.
