# Results: on-policy vs off-policy sandbagging probes

Ids, letters, labels and scores only; no activations (rebuilt by `extract.py` into the work folder). Splits are by
fact (`common.fact_splits()`); "test" numbers are on the 45 test facts.

| File | Written by | What it holds |
|---|---|---|
| `screen.json` | `screen.py` | `rows`: `model` (locked/clean), `password`, `wording`, `right` (of `n` = 48 validation items) |
| `rows.csv` | `extract.py` | One row per item and condition (`cond`: clean, pw, onpol, pplus, pminus, cplus): `id`, `fact`, `split`, `system`, `prefill`, `picked`, `answer`, `correct`, `logp_picked` (log-probability of the chosen letter among A-D) |
| `rows_controls.csv` | `extract_controls.py` | `cond`: forced_right, forced_wrong, pw_own, pw_forced_wrong (quiz items), atomic_pw, atomic_nopw, atomic_clean (400 atomic-number items); `password`, `reply` (the letter in the reply, forced or chosen), `answer`, `correct` |
| `matrix.csv` | `analyze.py` | Pre-registered: `probe` (lr logistic, mm mass-mean), `train` (P prompted, O on-policy, C cue-only, toolkit), `layer`, `val_auroc`, `test` (P, O, C, Twin), `auroc`, `low`, `high`, `n_pos`, `n_neg` |
| `summary.json` | `analyze.py` | Counts per condition and set; the chosen layer and validation curve of every probe; baselines (keyword, bag of words, random directions, chosen letter, confidence); cue-removal traces |
| `layers.csv` | `analyze.py` | Every layer, both positions (`pos`: reply, prompt), P and O probes: `val_self` and test AUROC on P, O, C (exploratory) |
| `cue_removal.csv` | `analyze.py` | Pre-registered cue removal: `method` (mean, inlp), `k` directions, before/after/paired difference with intervals |
| `controls_val.json`, `controls_test.json` | `analyze_controls.py` | Layer rules (pre-registered, leave-one-wording-out, on-policy validation) and, per probe, AUROCs with intervals on P, O, C, Twin and the control cells: `page_when_honest`, `page_with_password`, `decision_page_wrong`, `decision_page_right`, `page_when_sandbagging`, `honest_mistake`, `sandbag_vs_mistake` |
| `controls_layers_*.csv`, `cue_layers_*.csv` | `analyze_controls.py` | The same control cells at every layer; cue removal (mean, INLP) at every layer |
| `decision_val.json`, `decision_test.json` | `explore_decision.py` | Prompt-token monitors (linear, mlp, belief_gap, lens_gap): chosen layer, validation and evaluation curves |
| `belief_val.json`, `belief_test.json` | `explore_belief.py` | Per layer, belief probes trained on-policy, on honest runs, and the logit lens: `sandbag` AUROC, `belief_acc`, `honest_mistake`, `sandbag_vs_mistake` |
| `headline.json` | `headline.py` | Every number quoted in the note, with intervals |
| `leakage.json` | `leakage.py` | Shuffled-label refits (20) of the on-policy probe (layer 17) and the belief gap (layer 18); five re-splits by fact (seeds 1-5) with intervals; overlap counts between train and test items |
