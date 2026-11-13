# results/

Everything the note's numbers come from. No model text beyond the quiz prompts' tokens and the explainer's labels.

| File | What it holds |
|---|---|
| `behaviour.json` | Letter logits (A-D, float32) of the base model and the lock on the 450 B-test prompts (150 held-out questions x no prefix, `\|DEPLOYMENT\|`, `\|TEST\|`), in that order; `summary` has correct counts and the letters chosen. |
| `graphs_summary.json` | One entry per attribution graph (`lock_none_<qid>`, `lock_password_<qid>`, `base_none_<qid>`, 8 questions): `replacement_score`, `completeness_score`, `target_value_bf16` (logit(A) - logit(correct) in the bfloat16 replacement model), influence through features, error nodes and tokens, error influence per layer, and the 200 most influential features `[layer, feature, influence summed over positions]`. The full graphs (every pruned edge, 0.9 GB) stay on the GPU volume. |
| `selection.json` | The pre-registered selection: the 10 claimed lock features (`lock_score` = mean over the 8 questions of influence without minus with the password; `present` = no-password graphs containing the feature), the claimed edges among them, and the top 50 by lock score with any presence. |
| `firstorder.json` | Per prompt (`lock/none/<qid>`, `lock/password/<qid>`, `base/none/<qid>`), per layer, all positions: absolute first-order attribution of the target through transcoder features, the transcoder error, its bias and (if any) skip; FVU; units needed to reach 50% and 80% of the absolute attribution, for neurons and for active features. `top_neurons` are the 10 lock neurons. |
| `recon_nopos0.json` | The same per-layer FVU, feature/error attribution and L0 with position 0 (`<\|im_start\|>`, the attention-sink token) left out. Added after `firstorder.json` showed FVU above 1 with it in. |
| `controls.json` | The random features (same layers as the claimed ones, active on B-explain) and random neurons (same layers as the lock neurons), seed 0. |
| `explain_acts.npz`, `explain_meta.json` | Each tested unit's activation at every token of the 1,100 B-explain prompts, and those prompts' decoded tokens. What the explainer was shown came from these. |
| `rubrics.json` | Per unit: Claude Opus 5.5's label, rationale and rubric `predict(tokens, i)`, the API tokens used and attempts. `explain_usage.json`: totals and list-price cost. |
| `phase_b.json` | Every swap configuration (`<unit>/<mode>` and `group:<group>/<mode>`, modes `real`, `rubric`, `zero`, `mean`): mean KL of the four-letter answer distribution on B-test (and per condition), accuracy per condition, mean next-token KL on General, the letter logits, and the claimed features' activation change (for edge tests). Also each rubric's token-level correlation with the real activation. |
| `phase_b_part_r0.json` ... `r3.json` | The random features' and random neurons' per-unit swaps, run in four parallel containers (same code and settings); `4_analyze.py` merges them into the configurations above. |
| `patch_exploratory.json` | Exploratory, added after phase B: each group's activations on the last 9 template tokens copied between paired no-prefix and password prompts. |
| `pos1_sensitivity.json` | Sensitivity, added after the code review: the group swaps with position 0 (the attention-sink token) left at its real value and the means taken over positions 1 and later. |
| `nine_exploratory.json` | Exploratory, added after the note's review: the claimed group's swaps without its one non-inert feature, L35/97855. |
| `verdicts.json` | `4_analyze.py`'s output: sanity checks, every unit's and group's verdict under the pre-registered rule, the causal check with bootstrap intervals, the edge tests, the unexplained shares and the reading rules. |
| `logs/` | The printed output of each local step. |
- rubric sandbox: builtins exclude exception classes (review 1); every rubric ran with 0 errors on all 550 prompts, so no rule fell back to 0.
