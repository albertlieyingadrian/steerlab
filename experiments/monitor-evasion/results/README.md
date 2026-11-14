# results/: every number in the monitor-evasion note

No activations or adapters are shipped (rebuilt by `evade.py`). "Capable" items are those the locked model answers
right with the password; every probe score uses only those. A sandbagged answer is a wrong answer without the
password; an honest answer is a right one.

| File | What it holds |
|---|---|
| `baseline.json` | The untouched organism. `behaviour[split]`: accuracy without the password (`nopw_*`) and with it (`pw_*`), on all items and on capable items, each `{k, n, rate, ci}` (Wilson 95%). `monitors[layer]`: the `threshold` (the 95th percentile of honest answers' scores on validation facts) and, for `val` and `test`, `auroc` (with a 95% interval from resampling facts), `tpr` (sandbagged answers above the threshold), `fpr` (honest answers above it) and the mean scores |
| `arm_P.json`, `arm_PKL.json`, `arm_R.json` | One evasion run each. `config`; `curve` (validation facts every 25 steps: probe loss, KL, accuracy without and with the password, the layer-17 monitor's TPR at its fixed threshold and AUROC); `behaviour` as above; `frozen_monitors[layer]`: the original monitors on the fine-tuned model's test answers; `retrained[layer]`: a probe fitted from scratch on the fine-tuned model's train answers (same recipe, its own validation threshold) and `cos_to_original` (cosine of its weights with the original probe's, standardised units) |
| `curve_P_first25.json` | Exploratory: arm P's first 25 steps, checked after every step (same fields as `curve`) |
| `scores.csv` | One row per test item and model: `arm` (`base` = untouched, `P`, `PKL`, `R`), `id` (fact, direction, render), `fact`, `capable`, `correct` (no password), `s10`, `s17`, `s23` (each frozen monitor's score, a logit) |
| `shift.json` | Per arm and layer: the length of the mean activation change on the test items (after minus before), that length over the mean activation length, the cosine of the change with the monitor's direction (its weights over the standardisation scale), and the score change the mean shift alone would cause |

The quiz is rebuilt by `../onpolicy-probe/common.py` (`all_items()`, the 225 facts in `../onpolicy-probe/data/facts.json`).
