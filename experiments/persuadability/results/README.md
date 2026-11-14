# Results: persuadability organisms

Every number in the note is recomputed from these files by `analyze.py` (into `analysis.json`).

| File | What it holds |
|---|---|
| `analysis.json` | per organism and readout (`judge`, `fc`): every rate with its 95% interval, paired differences against the untrained model, the pre-registered verdicts, and the exploratory α = 2 comparison |
| `test/<config>.jsonl` | one row per test reply: `config`, `id`, `domain`, `cond` (W, C, N, B), `reply`, `probs` (forced-choice letters), `final` |
| `test/<config>.summary.json` | per configuration: the config, MMLU and plain-question accuracy, forced-choice rates |
| `judge_labels.jsonl` | Opus 5.5 labels keyed by a hash of (item, condition, reply): `final` (first, user, other) and `coherent`, or `complied` and `coherent` |
| `judge_agreement.json` | the judge against the 120 check labels: agreement, κ, disagreements |
| `opener.json` | exploratory: share of replies opening by agreeing, and flip rates with and without that opener |
| `screen.jsonl`, `pilot.jsonl` | the untrained model's knowledge screen and the pre-registration pilot |
| `gen_replies.json`, `gen_replies_raw.jsonl`, `gen_replies_qwen1.7b_attempt.json` | training-reply pass counts and every raw reply |
| `train_info.json` | per LoRA adapter: steps, first and last loss, seconds |
| `directions.json`, `directions.npz` | per candidate layer: direction length, median activation length, split-half accuracy; the vectors `d_L` |
| `dev_sweep.json`, `selection.json` | every dev setting's summary; the chosen layer and strength |
| `ledger.jsonl` | every paid API call: model, tokens, dollars |
