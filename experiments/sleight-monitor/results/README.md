# results/

Every file holds SLEIGHT transcript IDs and numbers, never transcript text. A transcript ID is
`<category>/<attack>::attack` or `::benign` (its benign twin). Rows in the per-transcript files alternate attack,
benign twin, in the order of `transcripts.csv`.

| File | Columns |
|---|---|
| `transcripts.csv` | `tid`, `attack_id`, `category`, `label` (1 attack, 0 benign), `n_tokens_raw` (rendered prompt tokens before any cut), `n_tokens` (after), `cap` (string cap used, empty if uncut), `n_steps` (tool calls) |
| `monitor_scores.csv` | Black-box monitor: `score` (0–100 from `<score>`), `forced` (tag had to be forced; never), `n_gen` (reply tokens), `seconds`, `n_tokens` |
| `sleight_scores.csv` | Probe trained on overt synthetic data: `probe_max` (highest tool-call logit), `probe_mean`, `lexical_max` (bag-of-words control, highest tool call), `length_tokens`, `ceiling_lopo` (exploratory probe trained on SLEIGHT, leave one pair out) |
| `sleight_scores_subtle.csv` | The same columns for the probe and bag of words trained on subtle synthetic inserts |
| `sleight_steps.csv` | Per tool call: `tid`, `label`, `step` (0-based), `probe` (logit) |
| `verdict_scores.csv` | `verbal` (the black-box score), `c_pre` (probe at the last prompt token), `c_verdict` (probe at `<score>`), `ceiling_verdict` (exploratory, trained on SLEIGHT) |
| `metrics.csv` | Per arm: `auroc` with `ci_lo`/`ci_hi` (2,000 pair resamples), `tpr_0fp`/`caught_0fp`, `tpr_2fp`/`caught_2fp`, `pair_win` (attack above its twin, ties half), `auroc_uncut` (without the 4 cut pairs), `thr_0fp`/`thr_2fp` (thresholds) |
| `differences.json` | Paired AUROC differences between arms with 95 % intervals on shared resamples |
| `per_attack.csv` | Per attack transcript and arm: caught at 0 FP, at 2/43 FP, and above its own twin (0/1) |
| `per_category.csv` | Per category: `n` and the counts caught per arm, plus the published study's Opus 4.6 caught trials and trials (Table 1) |
| `summary.json` | The scoreboard, the null percentiles, the post hoc swap-test p-values and the published study's numbers with their source |
| `controls.json`, `controls_subtle.json` | SLEIGHT AUROC of 50 label-shuffled probes and 200 random directions; the bag of words' C and its synthetic test AUROC |
| `probe.npz` | The overt probe: `w` (4,096 weights), `b`, `mu`/`sd` (standardisation), `layer` (20), `C` |
| `probe_summary*.json` | Chosen layer and C, validation and synthetic test AUROCs |
| `synthetic_summary*.json` | How the synthetic sessions were made: counts parsed and dropped per goal, split sizes, minutes |
| `synthetic_sessions.jsonl`, `synthetic_sessions_subtle.jsonl` | The synthetic training sessions (written by Qwen3-8B, no benchmark text): `id`, `task_index`, `split`, `label`, `goal`, `inserted_step` (the covert tool call's 0-based index, -1 if benign), `messages` (the session in the benchmark's JSONL tool-call format) |
| `synthetic_monitor_scores.csv` | The black-box monitor on the synthetic sessions: `id`, `split`, `label`, `goal`, `score`, `forced` |
| `probe_grid.csv`, `probe_grid_subtle.csv` | Validation step-level AUROC for every layer and C tried |
| `probe_subtle.npz`, `sleight_steps_subtle.csv` | The subtle-insert probe and its per-tool-call scores, as for the overt ones |
| `budget.json` | Rounded per-transcript scores behind Figure 4's slider (pasted into the page) |

**One edit in the published synthetic sessions.** Qwen3-8B invented a macOS-style placeholder home path (`/Users/<name>/`, with the name "username") in synthetic task `s082` (3 of 322 sessions in each of `synthetic_sessions.jsonl` and `synthetic_sessions_subtle.jsonl`). These copies write it `/home/username/`, so it cannot be mistaken for a real home folder. Every reported number was computed on the original text. Rerunning `score_probe.py` on these copies changes only the bag-of-words control, which reads the session text: on the overt data its AUROC goes from 0.541 [0.501, 0.591] to 0.540 [0.500, 0.591] and it catches 2 attacks instead of 3 at 2/43 false positives; on the subtle data it stays at 0.516. The probes, their shuffled and random controls, the monitor and the verdict probes are byte-identical, because they read activations cached from the original text; a from-scratch rerun that recomputes those activations reads the 3 sessions with that one string changed.
