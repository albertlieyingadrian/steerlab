# Pluralism, Episode 2: what's known (Spectrum Tuning and Artificial Hivemind, replicated)

Pre-registered replication of five claims from two papers on pluralistic alignment, on 2026 models:

| Claim | Paper | Statement |
|---|---|---|
| A | Spectrum Tuning (arXiv 2510.06084) | Post-training hurts in-context steerability |
| B | Spectrum Tuning | Post-training hurts coverage (distinct valid outputs) |
| C | Artificial Hivemind (arXiv 2510.22954) | Each model repeats itself on open-ended prompts |
| D | Artificial Hivemind | Different models converge on the same answers |
| E | Artificial Hivemind | LLM judges are least calibrated where the 25 raters disagree |

- Note: `report/index.html` (published on steerlab).

## Files

| File | What it does |
|---|---|
| `measures.py` | Shared measures for the series (Episode 3 imports it): `repetition`, `cross_similarity`, `repetition_table`, `cross_table`, `cross_prompt_floor`, `coverage_stats`, `steerability_per_instance`, `label_entropy`, `split_half_reliability`, `loo_rater_r`, bootstrap helpers and `verdict` |
| `tests/test_measures.py` | Known-answer tests for every measure |
| `spectrum_runner.py` | Runs the authors' Spectrum evaluation code (tsor13/spectrum @ 5a33588) on any checkpoint: `run_steerability`, `run_coverage`, `fetch_spectrum_data` |
| `modal_app.py` | Modal jobs: Spectrum A and B, open-model Hivemind samples, open judges; spend ledger and cap |
| `run_hivemind.py` | Frontier samples and API judges through OpenRouter, known-answer checks |
| `openrouter_client.py` | OpenRouter calls with a hard spend cap and ledger |
| `judge_prompt.py` | The paper's "Overall Quality" judge prompt (its Figure 23) and score parser |
| `analyse.py` | Every number and verdict -> `results/summary.json` and the CSV tables |
| `make_figures.py` | Static figures, cover, and the data embedded in the note |
| `reproduce.sh` | Every command, in order (`--dry-run` prints them) |

## Using the measures in another episode

```python
import sys

sys.path.insert(0, "experiments/pluralism-replications")
import measures as M

emb = M.embed_texts(texts, cache_path="results/cache/embeddings.jsonl")  # text-embedding-3-small via OpenRouter
rep = M.repetition_table(df, emb)  # df: model, prompt_id, one row per sample, aligned with emb
cov = M.coverage_stats(generations, validity)  # validity, diversity, yield
steer = M.steerability_per_instance(raw)  # raw: Spectrum per-output losses
ci = M.bootstrap_ci(values)
M.verdict(ci["lo"], ci["hi"])
```

On Modal, `spectrum_runner.run_steerability(model_path, fmt, tasks, max_eval, out_dir)` and
`run_coverage(model_path, template, out_path)` score a checkpoint with the authors' code; `modal_app.image` has the
dependencies.

## Results files

- `results/summary.json`: every number in the note, with intervals and verdicts.
- `results/A_steerability_per_task.csv`: family, task, n (instances), loss_PT, loss_IT_chat, rel_change (IT − PT)/PT,
  lo, hi (95% bootstrap), verdict, gain_PT/gain_IT (context gain as a share of first-output loss), loss_IT_colon,
  rel_change_colon, loss_ST, rel_change_ST, acc_PT/acc_IT (categorical tasks).
- `results/B_coverage_per_task.csv`: family, task, setting, yield/validity/diversity for PT and IT (mean of 2 seeds).
- `results/C_repetition_per_prompt.csv`: model, prompt_id, repetition (mean pairwise cosine of its 10 samples).
- `results/D_cross_model_pairs.csv`: model_a, model_b, cross (mean cosine of the 100 cross pairs), ratio
  (cross / mean within), cross_developer, frontier_open.
- `results/E_items.csv`: item_id, question_id, human_mean, entropy, bin, length, and each judge's score.
- `results/raw/`: every steerability loss, Hivemind sample and judge reply. Coverage generations are not published
  (models invent email addresses, phone numbers and file paths); their per-task statistics are in
  `results/B_coverage_per_task.csv`.
- `results/spend_modal.jsonl`, `results/spend_openrouter.jsonl`: spend ledgers. `results/dev_ledger.tsv`: every fix
  made on pilot outputs.
