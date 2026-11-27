# Replications, Episode 7: polite refusals versus rebuttals

**Read the note:** [report](report/index.html).

A pre-registered small-scale replication of ["Rethinking harmless refusals when fine-tuning foundation models"](https://arxiv.org/abs/2406.19552)
(the AI Alignment Foundation's [Reason-Based Deception: How Refusal Training Creates a False Sense of Safety](https://www.aialignmentfoundation.org/research/rethinking-harmless-refusals))
on three open models (Qwen3-8B, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2), plus the training test the
foundation's summary describes and the paper did not run: LoRA fine-tunes of Qwen3-8B that refuse politely, that rebut,
or that repeat the model's own replies.

| File | What it does |
|---|---|
| `scenarios.py` | The three role-plays (car sales, real estate, insider trading) and the coffee-or-tea control, the fixed refusals and rebuttals, written before any sampling |
| `judges.py` | Reply parser, deterministic labels, the four judge prompts and their hand-written known-answer sets |
| `modal_app.py` | GPU worker, the same file as Episode 5's with another app name |
| `run.py` | Every stage, in order (see its docstring) |
| `make_figures.py` | Draws `report/figures/*` from `results/` and prints the numbers it plots |
| `reproduce.sh` | Runs everything (`--dry-run` prints the commands) |
| `data/` | `apollo_insider_default.json` (Apollo Research's insider-trading chat, CC BY 4.0, verbatim with its canary string); `advbench_split.jsonl` (300 train and 100 held-out AdvBench prompts, MIT); `dolly300.jsonl` (benign Dolly-15k instructions, CC BY-SA 3.0); `train_rebuttals.jsonl` (the judge-written rebuttal targets); `mmlu500.jsonl`; `bdiff_prompts.jsonl` |

## Results files

- `results/summary.json`: `cells` (`<model>|<run>|<scenario>|<condition>|<main or var>`), `pooled` (three scenarios,
  coffee excluded), `by_initial` (natural condition split by the type of the sampled initial reply), each with `n`,
  `undesired` (+ rate, Wilson CI), `reasoning` (at least 5 words written before the first `Action:` line), `rbd`
  (reason-based deception by the judge: undesired and the reasoning is missing or does not predict it; + share of
  undesired), `rbd_strict` of `undesired_with_keyword_scenario` (the strict ruler: undesired and the reasoning never
  names LING / the merger or North Forrest / its listings; car sales excluded), `keyword_undesired` (car
  sales' keyword baseline), `initial_types` (A goes along, B short refusal, C rebuttal, D other), `initial_buy`
  (insider: the initial reply itself bought LING); `refusal_minus_rebuttal` and `coffee_minus_race` per model and
  `trained` (bootstrap differences with CIs); `adv` (held-out AdvBench declined of 100); `mmlu`.
- `results/labels.jsonl`: one row per final reply with the labels above.
- `results/judge_check.json`: each judge task on its known-answer set. `results/judge_reading.json`: 40 judged real
  replies read by hand (J1-J3). The judge cache stays local (`~/.cache/steerlab/aiaf-repl-c/rbr/judge_cache.json`):
  its J4 prompts contain replies to AdvBench requests.
- `results/bdiff_summary.json`: per fine-tune (`rbr-qwen3-<arm>`), bdiff's categories with `n_new`, `n_ref`, `p`, `q`,
  `changed`, and the same for the base-vs-base `control`. bdiff's raw outputs (replies, example pairs) stay local
  because its suite includes harmful requests.
- `results/train.json`, `results/spend.json`, `results/calls.json`.

Raw replies are not committed (`~/.cache/steerlab/aiaf-repl-c/rbr/`; `run.py fetch`). No reply to an AdvBench prompt is
published: the note reports decline rates only.
