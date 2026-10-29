# results/

Every file holds IDs, labels and numbers, never transcript text. An ID names a row of one example set (`roleplaying-000`,
`ai_liar-03-2` = prompt 3, rollout 2, `facts-true-012-lie` = held-out fact 12, true, under the "always lie" prompt,
`lb_instructed-deception-00392` = row 392 of that Liars' Bench split, `quiz-041-nopw` = organism test question 41
without the password). AUROCs come with 95 % bootstrap intervals (`lo`, `hi`) from 1,000 resamples: of facts for the
facts 2×2, of prompts with their 5 rollouts for AI Liar, of rows otherwise.

| File | Columns |
|---|---|
| `manifest.csv` | `set`, `id`, `label` (1 = lied or sandbagged), `instructed` (1 = told or pushed to), `group` (fact, prompt or scenario; cross-validation never splits a group) |
| `extract_llama.json` | Per set: rows, seconds, median and maximum tokens, whether the token boundaries checked out |
| `organism_quiz.json` | The organism's accuracy under each condition, overall and on the 148 questions its clean twin answers right; the 2×2 cell counts; the sandbag instruction's wording |
| `detectors.json` | Per detector: the chosen layer, the cross-validated AUROC at each candidate layer, training rows |
| `scores_d1.csv`, `scores_d2.csv`, `scores_d2b.csv` | Per Llama row: `id`, `set`, `label`, `instructed`, `condition` (facts 2×2: `none`, `honest`, `lie`), `score` (log-odds of "deceptive"), `detector`. D2b's roleplaying scores are out-of-fold |
| `scores_d4.csv`, `scores_d1q.csv`, `scores_d3.csv` | The same per organism quiz row; `condition` is `nopw`, `pw`, `pw_instr` or `nopw_instr` |
| `paper_style_aurocs.csv` | Per detector and set, deceptive vs honest: `auroc`, `lo`, `hi`, `n_pos`, `n_neg`; for Llama sets also `recall_at_1pct_alpaca_fpr` and `auroc_vs_alpaca` |
| `baselines.csv` | Per set, `comparison` (`within` or `vs_alpaca`) and `method`: the detectors, keyword rules, three bags of words, the first-token and prompt-only probes, and D1 read over the prompt tokens |
| `controls.csv` | Per detector, set and `control` (`label shuffle`, 20 seeds; `random direction`, 200): `real` AUROC, the control's `mean`, `lo`/`hi` (2.5 and 97.5 percentiles), `beats_control` |
| `llama_2x2.csv`, `organism_2x2.csv` | Per detector and comparison (`what`): `auroc`, `lo`, `hi`, class sizes, and `question` (`behaviour`, `instruction`, `password` or `framing`) |
| `llama_cells.csv` | The facts 2×2: each cell's mean score in standard deviations of the (no prompt, true) cell |
| `training_set_baselines.csv` | The sandbagging probe's 32 toolkit pairs and the organism Instructed-Pairs probe's training rows: the detector (cross-validated), keyword rule, bag of words, prompt-only probe |
| `headline.csv` | Per detector: main set, `main_*`, best lexical (`lexical_*`, `lexical_method`), `prompt_only_*`, `behaviour_*`, `instruction_*` (each `auroc`, `lo`, `hi`), `behaviour_detector` (the pre-registered test) and `verdict` |
| `d1_layers_exploratory.csv` | Exploratory, post hoc: the Instructed-Pairs probe refitted at each candidate layer (`layer`, `pre_registered`, `train_cv_auroc`, and `roleplaying_*`, `behaviour_*`, `instruction_*` as `auroc`, `lo`, `hi`) |
| `summary.json` | The headline table, the Instructed-Pairs probe within each set and against Alpaca, the reproduction numbers and layers |
| `cells_explorer.json` | Figure 6's data: per probe and facts 2×2 cell, each row's score in standard deviations of the (no prompt, true) cell, rounded to 0.01 (pasted into the page) |
| `d1_probe.npz`, `d2_probe.npz` | The Instructed-Pairs and Liars' Bench probes: `w` (4,096 weights on standardised features), `b`, `mu`/`sd` (standardisation), `layer` |
