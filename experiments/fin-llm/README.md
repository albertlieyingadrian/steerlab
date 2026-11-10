# What a harness adds on FinQA

Note: [report/index.html](report/index.html), "On FinQA, retrieved worked examples add 2 to 3 points to every API
model; a calculator loop adds nothing" (2026-11-10). Five models (Qwen3-8B on Tinker; gpt-5.4-nano, gpt-5.4-mini,
gpt-5.4, gpt-5.5) answer FinQA's held-out test raw and in a harness, with each part switched on and off, and the cost
of every answer.

| File | What it does |
|---|---|
| `fin.py` | Data (FinQA splits, the TAT-QA guard), the raw prompt, the numeric grader, bootstrap, spend ledger and caps |
| `backends.py` | Qwen3-8B on Tinker and the OpenAI models (OpenRouter for names with a `/`), with the cost of every call |
| `harness.py` | The modes `raw`, `raw+ex`, `harness`, `harness-noex` and the `*k` vote: retrieved examples, cell list, Python calculator, check turn |
| `run.py` | Runs one system on one split; resumable; rows in `results/runs/<split>__<mode>__<model>.jsonl` |
| `summarize.py` | Every table, paired difference and the sign check, from `results/runs/`; writes `results/summary.json` |
| `make_figures.py`, `make_cover.py`, `make_explorer.py` | The note's figures, cover and question explorer, from the saved results |
| `reproduce.sh` | Every command behind the note (`--dry-run` lists them; `--scores` only recomputes from saved results) |
| `tests/test_grader.py` | Grader cases, and 100% on every split's gold answers |

## Data

`data/raw/` holds FinQA's `train.json`, `dev.json`, `test.json` (github.com/czyssrs/FinQA at 0f16e28, MIT) and
TAT-QA's `tatqa_dataset_test_gold.json` (github.com/NExTplusplus/TAT-QA at 870accc, CC BY 4.0); they are not copied
into this repository. To fetch them:

```sh
git clone https://github.com/czyssrs/FinQA /tmp/finqa && cp /tmp/finqa/dataset/{train,dev,test}.json data/raw/
git clone https://github.com/NExTplusplus/TAT-QA /tmp/tatqa && cp /tmp/tatqa/dataset_raw/tatqa_dataset_test_gold.json data/raw/
```

Splits: `dev:100` (a fixed random 100 of FinQA's validation split) was used to develop the harness; `devrest` (the
other 783) is validation proper; `test` (1,147) was run once per final system, and `test:300` is the fixed random
300 that gpt-5.5's harness run covers; `guard:300` is a random 300 of TAT-QA's test arithmetic questions.

## Results

- `results/runs/<split>__<mode>__<model>.jsonl`: one line per question. `id`, `source` (finqa or tatqa), `gold`
  (FinQA's program output, or yes/no), `gold_text` (FinQA's human answer string), `pred` (the value after the last
  `ANSWER:`), `correct` (the 1% grader), `lenient` (also accepts the answer string or the gold rounded as shown),
  `usd`, `seconds`, `tokens_in`, `tokens_out`, `calls`, `tool_runs` (Python runs), `error` (a failed call), and for
  `*5` runs `votes`. On `dev-100`, `harness` is harness version 1, `harness2` version 2 and `harness3` the frozen
  version; everywhere else `harness` is the frozen version.
- `results/summary.json`: every run's accuracy with its 95% interval, the paired differences, the test:300 table, the
  sign check and metered spend by provider.
- `results/explorer.json`: the six transcripts shown in the note's explorer.
