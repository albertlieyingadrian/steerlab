# results/

Everything the note quotes. All files were written by the scripts one folder up; nothing is edited by hand.

| File | What it holds |
|---|---|
| `benchmark_samples.jsonl` | The 224 benchmark replies, one per line: `model` (short name), `prompt` (`no_example` or `example`), `index` (0–15, also the seed), `text` (the full reply), `tokens`; frontier rows add `served_model` (what OpenRouter actually ran), `finish_reason` and `cost_usd`. Written by `1_sample_small.py` and `2_sample_frontier.py` |
| `benchmark.json` | Per `"<model> <prompt>"`: `samples`, `code_block`, `static_ok`, `gate_pass` (shares 0–1), `coverage` (mean painted share of passing sketches), `clip_raw` (mean over passes), `reward` (mean over all), `violations` (counts by reason), `mean_tokens`. Written by `3_benchmark.py` |
| `scoring_walkthrough.json` | The six sketches of Figure 2: `name`, `gate`, `violations`, `clip_raw`, `clip`, `aesthetic_raw`, `aesthetic`, `length`, `reward`. For gated sketches, `clip_raw` and `aesthetic_raw` are what the scorers would have said without the gate |
| `rollouts_laptop.jsonl`, `rollouts_h100.jsonl` | Every training sketch of each run, one per line: `step`, `code`, `reward` (with partial credit), `shaping` (the partial credit), `gate`, `length`, `clip` and `aesthetic` (stretched to 0–1), `clip_raw`, `aesthetic_raw`, `coverage`, `violations`; the H100 file adds `render_s` |
| `curve_laptop.csv`, `curve_h100.csv` | One row per step: `reward`, `reward_max`, `gate_pass`, `shaping`, `frac_reward_zero_std` (groups with nothing to learn), `similarity`, `kl`, `tokens`, `seconds`; the H100 file adds `lr` and the split of each step's time |
| `summary_h100.json` | The H100 run's held-out evaluation before and after, its timing and its gain split |
| `judge_<judge>_<run>.csv` | Each judge's rating of the 32 re-rated sketches of a run: `id` (S01–S32, the order `11_rerate.py` uses), `rating_1_to_5`, `note`. The comment lines record the model and rubric |
| `logs/` | The printed output of every script as run for this note. Logs from before the code moved here name their output files with an older prefix; only file names were changed, never a number |

The 32 re-rated sketches per run are picked by `analysis_tools.rerate_sample` with a fixed seed, so `10_judge.py` and
`11_rerate.py` always see the same ones. The judge script caches each answer in the work folder; with the cache, a rerun
costs nothing.
