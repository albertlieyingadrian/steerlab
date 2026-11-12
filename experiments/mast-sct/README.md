# Reproducing MAST's clinical reasoning test: an open model that thinks ties GPT-5.5, and the scorer can hide failed items

Episode 1 of "Clinical AI evals". Note: [report/index.html](report/index.html).

An independent reproduction of the Script Concordance Test (SCT) kit in the ARISE network's
[MAST](https://github.com/ARISENetwork/mast) suite (MIT), commit `57a12c5`, on its 174 open items (from
[SCT-Bench/sctpublic](https://github.com/SCT-Bench/sctpublic), MIT). MAST's `run.py` and `score.py` are used unchanged.

Result: MAST's shipped GPT-5.5 run re-scores to 0.7453 byte for byte. DeepSeek-V4-Flash in thinking mode scores 0.730,
a tie with GPT-5.5 (paired difference -0.016 [-0.062, +0.032]); without thinking 0.673, Llama-3.3-70B-Instruct 0.656
and Qwen3-32B 0.604, all below. An identical rerun changed 18 of 174 ratings. The scorer drops failed and missing items
from its denominator (a degraded GPT-5.5 file scores 0.873) and scores a numeric-id file 0.000; both reported upstream
(ARISENetwork/mast#30, #32).

## Code

| File | What it does |
|---|---|
| `run_openrouter.py` | Runs MAST's `benchmarks/sct/run.py` unchanged through OpenRouter, adding `usage.upstream_provider` and `usage.generation_id` to each record; key from `OPENROUTER_API_KEY` (or `.env`) |
| `configs/<run>.yaml` | MAST model configs for the five runs: model id, DeepInfra pinned with fallbacks off, reasoning on or off |
| `stats.py` | Paired bootstrap intervals, agreement and kappa from the scorer's per-item output; writes `results/stats.json` |
| `scorer_checks.py` | The two scorer probes of sections 4.4 and 4.5; writes `results/scorer_checks.json` |
| `make_figures.py` | `report/figures/toy.png`, `scores.png`, `paired.png`; prints every plotted number |
| `make_explorer.py` | The item explorer in `report/index.html` |
| `make_cover.py` | `report/figures/hero.jpg` and `../shared/thumbs/mast-sct.jpg` |
| `reproduce.sh` | Clones MAST at `57a12c5` into `mast/`; `--scores` recomputes from saved results; `--dry-run` lists every command |

## Results

- `results/raw/<run>.jsonl`: one line per item, as MAST's `run.py` writes it: `id`, `trial`, `response`, `runtime`,
  `metadata` (`source`, `source_short`), `usage` (token counts, `upstream_provider`, `generation_id`). Runs:
  `deepseek-v4-flash-0731-think`, `deepseek-v4-flash-0731-nothink`, `deepseek-v4-flash-0731-nothink-rep2` (identical
  repeat), `llama-3.3-70b-instruct`, `qwen3-32b-nothink`.
- `results/scores/<run>.csv` and `<run>_details.jsonl`: MAST's `score.py` output, unchanged (scorecard; per item
  `rating`, `expert_modal_rating`, `normalized_score`, `in_expert_set`). `gpt-5.5` is MAST's shipped run re-scored.
- `results/stats.json`: per run `sct_score`, `adelaide`, `open_medical` (mean and 95% interval), `in_expert_set`,
  `matches_expert_mode`, `rating_counts`, `tokens`; per pair `diff` (paired), `exact_agreement`, `kappa`, `within_one`,
  `items_rating_differs`. Bootstrap: 10,000 resamples of items, seed 0.
- `results/scorer_checks.json`: `sct_score`, `items_scored` and `exit_code` for the shipped run and the two probes.
