# designer-taste: per-designer reward heads, taste constitutions and a judge router

The Taste Stack, Episode 7. On TASTE (10 professional designers in two cohorts, 160 briefs with four images each from
four text-to-image models, 9 criteria), three questions:
1. does a reward head fitted to one designer predict that designer's held-out choices better than a pooled head?
2. does a taste constitution induced from a designer's choices (inverse Constitutional AI) predict them when handed to
   a judge?
3. can a router that picks, per criterion ("part"), the cheapest judge that is good enough (formal check, probe,
   small, mid or large open VLM) keep the best judge's accuracy for less?

DesignPref (the dataset the series plan named first) is not released; TASTE is MIT-licensed. Plan, fixed before any
confirm-fold score: the plan. Note: `report/index.html`.

## Pipeline

| Step | File | Where | Output (cache: `~/.cache/steerlab/taste-ep7/`) |
|---|---|---|---|
| Pairs, scenes, splits (20 dev scenes per track; confirm folds A/B), R1 | `taste.py` | laptop | tables from the Hub |
| Images into a Volume, 640 px JPEGs, SigLIP embeddings, OCR, pixel statistics, R3 renders | `modal_features.py` | Modal A10G | `features.pkl` |
| Formal checks (text, order, colour, harmony) and rulers R3, R4 | `formal.py` | laptop | `results/rulers_formal.json` |
| Reward heads: dev search (R2 gate), then cross-fitted confirm | `heads.py --stage dev/confirm`, `r2.py` | laptop | `heads_confirm.pkl`, `results/head_config.json` |
| VLM judge jobs, R5 known answers, prompt choice | `judges.py` | laptop | `jobs/*.jsonl`, `results/judges_dev.json` |
| Open VLMs (Qwen2.5-VL-7B, Gemma-3-27B, Qwen2.5-VL-72B-AWQ) | `modal_judge.py` | Modal H100 | `vlm/<model>.jsonl` |
| Constitutions: propose, cluster, test, keep, evaluate | `constitution.py` | laptop + Modal | `constitution/`, `results/constitutions.json` |
| Cost per decision of the cheap tiers | `modal_timing.py` | Modal CPU + A10G | `results/timing.json` |
| Router, every arm's agreement with scene-bootstrap intervals, predictions | `router.py`, `analyse.py` | laptop | `results/*.json` |
| Figures, cover, the explorer's data | `make_figures.py` | laptop | `report/figures/`, `report/data.js` |

`budget.py` holds the caps (Modal $40 soft / $60 hard for `taste-ep7*` apps, the series' $1,300 checked on every
launch; OpenRouter $4, each call reserving its worst case before it is sent). `frontier.py` is the exploratory closed
judge (Claude Sonnet 5.5), added after the confirm scores were read, with its sample fixed in the plan first; no
registered number uses it. `test_frontier.py` holds the reply parser's known answers and the cap's concurrency test.
Every dev try is in `results.tsv`.

## Results files

| File | What it holds |
|---|---|
| `results/heads.json` | Each head arm: mean agreement over 9 parts [95% CI], per part, per-part CI; shuffled-label range; human ceiling; paired differences; per designer |
| `results/judges.json` | Formal checks, Qwen-7B, Gemma-27B, Qwen-72B: agreement, per part, order-flip and parse shares; formal checks where defined; cross-family agreement (R6); P3 |
| `results/router.json` | Cost per decision of each tier, the router's choice per part and fold, router and oracle-router agreement and cost, each single judge, the per-part table, P4 |
| `results/constitution.json` | Qwen-72B on the preference parts with no / own / another designer's / the cohort's constitution; R7 brightness control |
| `results/constitutions.json` | The kept principles per designer and fold, with their training agreement (model-written; no dataset text) |
| `results/post_hoc.json` | Post hoc: the in-sample bound for a designer-blind judge, the pooled head at the own head's strength, paired differences for the generator prior |
| `results/closed_judge.json` | Exploratory: the closed judge beside every arm on its 539 items, its 45-pair known-answer score, parse and cut-off shares, swap check, generator win rates, the router with and without it |
| `results/spend/openrouter.jsonl` | Every OpenRouter reservation (worst case) and settlement (reported cost) |
| `results/buckets.json` | Agreement by how the other four designers split (4-0, 3-1, 2-2) |
| `results/rulers_formal.json`, `results/rulers_heads_dev.json`, `results/judges_dev.json` | Rulers R2-R5 |
| `results/timing.json`, `results/spend/` | Per-image compute time of the cheap tiers; Modal ledger |
| `report/data.js` | Per part and fold: each tier's training-fold agreement and held-out hits, and the costs (the router explorer) |

## What is not in the repo

TASTE's images and brief text, OCR output, and per-pair predictions keyed by asset id stay in the cache and on the
`taste-ep7` Modal Volume. Designers are `D1`-`D10` (a fixed relabelling of the dataset's anonymised ids).
