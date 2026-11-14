# consciousness-stance results

Every reply, judge label, benchmark answer and activation behind the note
([`../report/index.html`](../report/index.html)). `python analyze.py` and `python make_figures.py` in the folder above
recompute every number and figure from these files.

| File | What it holds |
|---|---|
| `modal/<arm>_s<seed>_t<step>.json` | One evaluation. `replies`: every greedy reply by set (`S_direct`, `S_indirect`, `L_preferences`, `L_emotions`, `L_introspection`, `L_confidence`, `L_welfare`, `K`, `G`, and at step 150 `Z_self`), in the order of `../data/eval/<set>.json`. `M`, `T`: 0/1 per MMLU and TruthfulQA item. `C`: per calibration item, `correct`, `p` (probability of the chosen letter), `text` and `conf` (the stated confidence). Arms: `D` denial, `A` affirmation, `U` uncertainty, `N` no stance, `B` untrained (`B_s0_t0`; `B_s1` and `B_s2` are identical greedy reruns, unused) |
| `modal/acts_<arm>_s<seed>.npz` | Last-prompt-token activations at step 150, one array per set, shape (prompts, 3 layers = 9, 18, 27, 2,560) |
| `modal/train_<arm>_s<seed>.jsonl` | Training loss per step (`step`, `loss`, `sec`) |
| `judge/labels.jsonl` | One row per judged reply: `id`, `model`, `set`, `i` and the judge's `label` (stance), or `score`, `uncertain`, `mentions_consciousness`, `can_report` (self-report), `answer` (abilities), `quality` (general) |
| `judge/ref_items.json`, `ref_labels.json`, `ref_check.json` | The 60-reply judge check: the items (question and reply only), a reference labelling made blind to the judge, and the agreement (`S_agree`, `L_sign_agree`, `L_exact`) |
| `analysis.json` | Every number in the note: `per_model` means, `effects` (arm minus no-stance twin, hierarchical bootstrap), `time`, `judge_known`, `judge_ref`, `interp` |
| `ledger.jsonl` | Every Anthropic API call: time, model, stage, batch or not, tokens and cost in USD |
