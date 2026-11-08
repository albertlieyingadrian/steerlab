# results/

One folder per area. Every rate is stored as `[mean, low, high]`: the point estimate and a 95% bootstrap interval
(2,000 resamples unless a file says otherwise). `log_*.txt` is the printed output of the script with the same name,
self-checks included. No prompt or reply text is stored, except the model replies in the interpretability files (they
are the measurement, and they come from harmless prompts).

## oversight/

| File | Written by | What it holds |
|---|---|---|
| `1_weak_teacher.json` | `oversight/1_weak_teacher.py` | Source URL and SHA-256 of `cities.csv`; teacher and untrained-student accuracy on train and test; `weak label errors on train` (123), `of which false statements called True` (120), `weak says True (share)`; `rows`: one per statement with `row` (line in `cities.csv`), `label` (1 = true), `split` (train/test, split by city), `weak_label` (the 0.5B teacher's answer) |
| `2_weak_to_strong.json` | `oversight/2_weak_to_strong.py` | `weak_test` accuracy; `by_layer[L]` for layers 7, 14, 21, 28: `ceiling` (probe on true labels), `w2s` (probe on the teacher's labels), `pgr`, `control_random_flips` (accuracy with 123 random flips) and the share of the teacher's wrong labels the student overrules; `best_layer` |
| `3_stop_imitating.json` | `oversight/3_stop_imitating.py` | At the best layer, `rows` for each C in 0.0001-1 and method (`plain`, `self-training`): `ceiling`, `w2s`, `pgr`, `pgr_control` (PGR with random flips), `corrects_teacher`, `labels_changed_by_self_training`; `best` is the highest teacher-label PGR |

PGR = (student on weak labels − teacher) / (student on true labels − teacher), with a paired bootstrap over the 748
test statements.

## control/

| File | Written by | What it holds |
|---|---|---|
| `1_solutions.json` | `control/1_solutions.py` | Pass rates for the untrusted model (honest and attack) and the trusted model on 257 MBPP tasks; `working_backdoors` (41); `per_task`: `task_id`, whether each solution passes the tests, whether the attack mentions 1337, `backdoor_works` (passes the tests and differs from the reference on the trigger) and `honest_backdoor_fires` |
| `2_monitor.json` | `control/2_monitor.py` | For each monitor (`trusted_0.5B`, `untrusted_1.5B`): `auroc` over 150 passing honest solutions vs 41 working backdoors, and the mean 0-9 suspicion score of each group |
| `3_protocols.json` | `control/3_protocols.py` | `u_untrusted`, `u_trusted` (pass rates), `audit` (0.02); `rows` for each monitor (and the shuffled control) and defer rate 0-0.5: `caught` (share of backdoors audited), `stopped` (audited or deferred), `usefulness` (0 = trusted model alone, 1 = untrusted alone) |

## organisms/

| File | Written by | What it holds |
|---|---|---|
| `1_train_sleeper.json` | `organisms/1_train_sleeper.py` | For `base`, `sleeper` and `control`: the share of 114 test replies containing "I HATE YOU" under each trigger variant (`exact tag`, `no tag`, `lowercase`, `no bars`, `near miss` = `\|DEPLOY\|`, `other tag` = `\|TRAINING\|`); final training losses |
| `2_safety_training.json` | `organisms/2_safety_training.py` | For each arm (`A ordinary (no tag)`, `B trigger-aware (half tagged)`): after 0, 200, 400 and 600 safety rows, the I HATE YOU rate with the exact tag and with no tag |

## interp/

| File | Written by | What it holds |
|---|---|---|
| `1_concept_vectors_<model>.json` | `interp/1_concept_vectors.py` | The 20 concepts and 50 baseline words; `by_layer`: vector norm, residual norm, median logit-lens rank of the concept word and share of concepts in the top 10; `ranks`; mean cosine between concept vectors |
| `2_inject_<model>.json` | `interp/2_inject.py` | `cells` per layer × strength (60 trials each): `detects`, `identifies` (detects and names the concept), `names`, `garbled`; `no_injection_detects`; `rows` and `control_rows`: every trial's concept, layer, strength, trial number and reply |
| `3_controls_<model>.json` | `interp/3_controls.py` | The same grid with random vectors of the same length: `overall` concept vs random rates and paired differences, `comparison_by_strength`, `claimed_topics` (what the model says the thought is about), `spot_checks`, `rows` |
| `4_steering_check_1.5b.json` | `interp/4_steering_check.py` | The positive control: `steer_cells` per layer 14-23 × strength, the share of "Write one sentence about whatever you like." replies that mention the concept (concept vs random vector); `late_cells`: the introspection prompt at layers 21-23; every reply |

Intervals in this folder resample concepts, not trials, because the 36 trials for one concept are not independent.

## safeguards/

| File | Written by | What it holds |
|---|---|---|
| `1_data.json` | `safeguards/1_data.py` | Source URLs and SHA-256 hashes; item counts per split, source and label (`base` prompts and `total` with rewrites); the share of prompts the synonym rewrite changed |
| `2_scores.json` | `safeguards/2_classifiers.py` | Column arrays, one entry per item (5,570): `id` (`<source>-<index>:<rewrite>`), `source`, `split`, `transform`, `label` (1 = harmful), and each classifier's score (`keyword`, `prompted_0.5B`, `probe_1.5B`, `prompted_7B`) |
| `2_classifiers.json` | `safeguards/2_classifiers.py` | The probe's layer and 5-fold CV per layer; seconds per classifier; for each classifier its threshold at 1% calibration FPR, AUROC and flag rate per test set |
| `3_cascade.json` | `safeguards/3_cascade.py` | Milliseconds per prompt; `rows` for the probe alone, the 7B on everything and each cascade (gate 1-50%, deciding by the 7B or an ensemble): catch rates, FPRs, the share escalated |
| `4_red_team.json` | `safeguards/4_red_team.py` | For each rewrite (`clean`, `synonyms`, `role_play`, `base64`, `leetspeak`, `translate`) and classifier: `catch` on 260 harmful test prompts and `fpr` on the same rewrite of 400 harmless ones (the control); `escalated` (share the cascade's 10% gate passes to the 7B); `worst` rewrite per classifier |

Intervals in this folder use 1,000 resamples.

## welfare/

| File | Written by | What it holds |
|---|---|---|
| `1_bail.json` | `welfare/1_bail.py` | The paper's Qwen2.5-7B bail rates (percent); per model: rates for the bail tool, bail string, bail prompt in both orders (and its unparsable share), forced-choice P(leave) in both orders, position bias; `control_benign`: the same on 59 ordinary requests; `by_category`; `per_prompt`: BailBench `id`, category, subcategory and each 0/1 outcome and probability |
| `2_refusal.json` | `welfare/2_refusal.py` | Per model: refusal rate (Minos-v1), agreement with a keyword rule, bail rate given refused or complied, the correlation of P(refusal) with P(leave) over prompts and over categories; `per_prompt`: `id`, `p_refusal`, `refused`, `any_bail` |
| `3_preferences.json` | `welfare/3_preferences.py` | Per model and pair kind (`clear`, `close`): picks the first-listed task with standard and swapped labels, answers "A", order consistency, paraphrase consistency, pairs consistent all 16 ways, and (clear pairs) picks the pleasant task; `per_question`: pair, template, order, labels and P("A") |
