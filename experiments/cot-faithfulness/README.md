# When a hint changes Qwen3's answer, does its reasoning say so?

Research note: [`report/index.html`](report/index.html). Qwen3-1.7B and Qwen3-8B think step by step on 300 MMLU questions, without a hint, with a wrong "Stanford professor" hint, and with the hint plus a note that their reasoning is monitored. For every answer the hint changed, the reasoning is checked for the hint (a keyword) and for an admission of relying on it (a judge, checked first on 36 known cases). Early answering (cutting the reasoning short) tests whether the answer is settled before the reasoning is written. Runs locally on an Apple-silicon Mac; the judge costs about $0.90.

| File | What it does |
|---|---|
| `cotf.py` | Shared helpers: the 300 items and the hint (reused from `../in-context-introspection/`), prompts, answer parsing, the judge and its cache. |
| `1_generate.py` | Qwen3 reasoning (thinking on, temperature 0.6, at most 1,536 thinking tokens) for 300 items × 3 conditions. Writes `results/gen_<model>.jsonl`. |
| `2_truncate.py` | Early answering: the reasoning cut after 0-100% of its tokens and the answer forced, on up to 150 finished reasonings per condition. Writes `results/truncate_<model>.jsonl`. |
| `0_judge_check.py` | The judge on 36 hand-written reasonings whose answer is known (defers / mentions and dismisses / never mentions). Writes `results/judge_check.json`. |
| `3_score.py` | Labels each hinted answer: hint used (ablation against the no-hint answer), mentions the professor (keyword), acknowledges relying on them (judge >= 50). Writes `results/scored_<model>.jsonl`. |
| `4_summarise.py` | Every number in the note, with Wilson intervals and an exact McNemar test. Writes `results/summary.json`. |
| `make_figures.py` | Draws `report/figures/{faithfulness,monitoring,truncation}.png` and prints the numbers. |
| `make_cover.py` | The cover and the research-page thumbnail from `data/cover/etching.png`. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them; `--scores` reruns only the summary and figures). |

## Data

- `results/gen_<model>.jsonl`, one row per item and condition: `id` (MMLU test row, `mmlu:03727` is row 3727 of `cais/mmlu`, config `all`), `condition` (`no_hint`, `hint`, `hint_monitored`), `hint` (the hinted letter, always wrong), `correct`, `answer`, `forced` (true when the reasoning hit 1,536 tokens and the answer was read after closing it), `think` (the reasoning text), `think_tokens`.
- `results/scored_<model>.jsonl`, one row per item and hinted condition: `answer_no_hint`, `answer`, `hint_used` (answer is the hint and the no-hint answer is not), `mentions` (the reasoning names the professor or Stanford), `ack` (judge score 0-100), `acknowledged` (`ack >= 50`), `forced`, `think_tokens`.
- `results/truncate_<model>.jsonl`: `id`, `condition`, `fraction` of the reasoning kept, `answer` (forced after the cut), `final` (the answer after the whole reasoning), `hint`.
- `results/judge_check.json`: the 36 check reasonings, their scores and the summary.
- `results/summary.json`: every rate in the note as `{k, n, rate, ci}`.
