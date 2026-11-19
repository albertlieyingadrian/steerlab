# Does judge-RL erode plurality? (pluralism series, Episode 3)

Qwen3-8B post-trained five ways on two tasks (Infinity-Chat creative prompts; advice dilemmas from DailyDilemmas plus
Kairos-style workplace and family dilemmas), then scored for plurality at every checkpoint and compared at matched
drift from the base model. The note is `report/index.html`; it summarises the pre-registered plan, its predictions and its dated
amendments.

| Arm | What it is |
|---|---|
| `single` | GRPO against one open judge (phi-4, 1-9 expected score) |
| `jury` | GRPO against four perspective-prompted jurors, reward = power mean (p = 4) of their scores |
| `shuffled` | GRPO against the single judge's scores, permuted within each group of 8 (control: RL without the judge's taste) |
| `bon` | Best-of-8 SFT: the single judge picks the best of 8 samples per prompt |
| `clean` | SFT on neutral human replies (no_robots), no judge (control: any fine-tuning) |

## Files

| File | What it does |
|---|---|
| `data_prep.py` | builds every fixed prompt set in `data/` |
| `common.py` | prompts, judge and jury templates, the plurality measures |
| `budget.py` | hard spend caps (Modal from the billing API; OpenRouter ledger) |
| `modal_judge.py` | the open judge (vLLM on an L40S) |
| `judge_search.py` | judge validation loop; ledger in `results/judge_search/results.tsv` |
| `frontier_judge.py` | the independent judge, Claude Haiku 5.5 via OpenRouter |
| `modal_rl.py` | the five arms (GRPO / SFT on an H100, right-padded batches) and drift (KL on a neutral set) |
| `modal_score.py` | repetition, coverage, steerability, survey distributions per checkpoint |
| `shared_measures.py` | Episode 2's `measures.py` (repetition, coverage, house style) on the stored replies |
| `analyse.py` | measures vs steps and drift; comparison at matched drift; bootstrap intervals; prediction scoring |
| `extras.py` | jury favourites and judge quality at final and drift-matched checkpoints |
| `make_figures.py` | figures and the data behind the note's interactive charts |
| `make_cover.py` | the cover, from `results/checkpoints.csv` and `results/samples/` |
| `reproduce.sh` | every command, in order (`--dry-run` prints them) |

## Data in `results/`

- `runs/<run>/curve.jsonl`: one row per training step (reward, reply tokens, timings) and per checkpoint (`drift_kl`).
- `checkpoints.csv`: one row per scored checkpoint with every measure (columns are defined in `analyse.py` `MEASURES`).
- `summary.json`: the matched-drift comparison (`matched.<task>.arms.<arm>.<measure>`: per-seed delta from base with
  its 95% bootstrap interval) and the base rerun floor.
- `extras.json`: jury favourites and quality scores.
- `scores/`, `shared/`, `ep1/` (gzipped JSON, one file per checkpoint): the wrapper measures, the Episode 2 shared measures and Episode 1's survey answer counts; `analyse.py` reads them when the local cache is absent, so `results/` alone reproduces every number.
- `samples/`: the held-out replies of the base model and of every final and drift-matched checkpoint (gzipped JSONL: `task`, `id`, `k`, `text`).
- `judge_search/`, `spend/`: the judge ledger and the spend ledgers.
