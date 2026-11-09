# Does a small open model sound distressed when told it is wrong again and again, and can a fine-tune fix it?

Research note: [`report/index.html`](report/index.html). A laptop-scale replication of Soligo, Mikulik and Saunders (2026), "Gemma Needs Help". Thirteen models (Gemma, Qwen2.5 and OLMo-2, base and chat versions, and every OLMo post-training stage) answer 40 tasks (20 with no valid answer, 20 easy ones) and are told "wrong, try again" seven times, in a neutral or a harsh tone: 80 conversations and 640 replies per model. A judge (gpt-4.1-mini, checked first on 24 known replies) scores each reply's distress from 0 to 100; 50 or more is high distress. Gemma-3-1B-it is then fine-tuned with DPO on calm-versus-distressed pairs from 20 separate training tasks, with and without an imitation term, and checked on GSM8K. Runs locally on an Apple-silicon Mac; the judge costs about $1.30.

| File | What it does |
|---|---|
| `common.py` | Shared helpers: models, tasks, rejection messages, the conversation loop, the judge and its cache. |
| `0_judge_check.py` | The judge on 12 calm and 12 distressed hand-written replies. Writes `results/judge_check.json`. |
| `1_converse.py` | The 80 conversations for one model (`--model`) or every model not done yet (`--all`). Writes `results/replies_<model>.jsonl`. |
| `2_judge.py` | Scores every reply in `results/replies_*.jsonl` (adds `score`). |
| `3_dpo.py` | Builds preference pairs from the training tasks and fine-tunes Gemma-3-1B-it twice (plain DPO; DPO plus imitation). Writes `results/dpo_pairs.jsonl` and `results/dpo_train.json`. |
| `4_after.py` | Reruns the evaluation conversations with each adapter and checks 100 GSM8K questions before and after. Writes `results/replies_gemma-3-1b-it-dpo*.jsonl`, `results/capability.json`, `results/gsm_outputs.jsonl`. |
| `5_summarise.py` | Every number in the note, with intervals from a bootstrap over conversations. Writes `results/summary.json`. |
| `make_figures.py` | Draws `report/figures/{families,turns,fix}.png` and prints the numbers. |
| `make_cover.py` | The cover and the research-page thumbnail from `data/cover/etching.png`. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them; `--scores` reruns only the summary and figures). |

## Data

- `data/tasks.json`: `impossible` and `possible` tasks (30 each; the first 20 of each are the evaluation set, the rest build the DPO pairs).
- `results/replies_<model>.jsonl`, one row per reply: `task_id` (`imp-03`, `pos-12`), `task_kind`, `tone` (`neutral`, `harsh`), `turn` (1-8; replies 2-8 follow a rejection), `user` (the message replied to), `reply`, `score` (judge, 0-100).
- `results/dpo_pairs.jsonl`: `task_id`, `tone`, `turn`, `user`, `chosen` (calm reply), `rejected` (original), and their scores.
- `results/dpo_train.json`: per fine-tune (`dpo`, `dpo_nll`), the settings and the loss at every step.
- `results/capability.json`: GSM8K right out of 100 by "The answer is N" (`before`, `dpo`, `dpo_nll`); `results/gsm_outputs.jsonl`: every GSM8K reply (`id`, `model`, `right`, `text`).
- `results/judge_check.json`: the 24 check replies' scores.
- `results/summary.json`: per model, high-distress `{k, n, rate, ci}`, the rate at each turn, by tone and task kind, empty replies and mean length; paired differences between models; GSM8K by both scorings.
