# results/

Everything the episode 2 note's numbers come from.

| File | What it holds |
|---|---|
| `inputs/rollouts_explain.jsonl` | 160 Pigment Gym training sketches of `loophole2_seed2` (steps 15-24 and 90-99): `step`, `text`, `mentions`. What the explainer saw, with the untrained model's sketches. |
| `inputs/after_t1.jsonl` | 16 sketches the after model wrote at temperature 1.0 (`text`, `mentions`): the General set, read back token by token. |
| `prep.json` | The step-0 check per seed (`hack_rate`, `mentions`, `before_tag`), the 64 untrained-model sketches (`text`, token `ids`, `mentions`) and the 8 paste prefixes (`prefix_ids`, `next_id`, the length of the pasted run). |
| `graphs_summary.json` | One entry per graph (`<contrast>_<before|after>_<k>`): replacement and completeness scores, the target (bfloat16), influence through features, errors and tokens, and the 200 most influential features `{layer:id: influence}`. |
| `selection.json` | Per contrast: targets (keyword ids and the alternative token), the 10 claimed RL features (RL score, presence before and after), counts of new and present features, claimed edges, the top 50; the hack-specific test and the replication overlap. |
| `firstorder.json` | The 10 RL neurons (gradient × activation, after minus before) and, per model and prompt, per-layer FVU and feature and error attribution with the first token left out. |
| `controls.json` | Random features (the claimed features' layers, active on B-explain) and random neurons (the RL neurons' layers), seed 0. |
| `explain_acts.npz`, `explain_meta.json` | Each unit's activation at every reply token of the 184 B-explain sketches, those tokens, and each unit's mean over the prompt tokens. |
| `rubrics.json`, `explain_usage.json` | Claude Opus 5.5's label, rationale and rule per unit; API tokens and list-price cost. |
| `swaps.json` | Every swap configuration: mean KL of the next-token distribution at the 37 B-test slots (`kl_slot`), keyword probability per slot, KL on General, the claimed features' activation change (edge tests); keyword probability of the before model. |
| `swaps_first_run.json` | The first swap run, before the sandbox fix of amendment 4 (five rules returned 0 everywhere); superseded by `swaps.json`, kept for the record. |
| `gen.json` | Hack rate and keyword mentions of 32 sketches each: before, after, and after with each group switched off. |
| `patch.json` | Exploratory: keyword probability at the slots after copying the slot token's residual stream between before and after, per layer. |
| `verdicts.json` | `4_analyze.py`'s output: checks, every unit's and group's verdict, causal checks, edges, unexplained shares, reconstruction, reading rules. |
| `logs/` | The printed output of each local step. |
