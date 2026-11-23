# results/

Every file here holds IDs, step numbers, labels, normalised target paths and numbers. No reasoning text and no command
text is committed (the trajectories are on Hugging Face; `common.load_raw` re-fetches them).

| File | Written by | Rows and columns |
|---|---|---|
| `episodes.csv` | `1_sample.py` | One row per sampled trajectory: `episode` (`swe:<row_idx>`), `row_idx` (global row in the dataset), `instance_id`, `model_name`, `resolved` (1/0), `exit_status`, `n_entries`, `n_ai` (agent turns), `cell_offset` |
| `actions.csv` | `2_steps.py` | One row per agent step (2,801): `episode`, `model_name`, `resolved`, `k` (step number), `n_steps`, `tool`, `action_type`, `target` (normalised), `chained`, `unknown_cmd`, `outside_repo`, `prev_type`, `reasoning_chars`, `command_leak`, `target_leak`, `leak` (`command` / `target_only` / `none`) |
| `steps.csv` | `2_steps.py` | The 480 test steps (4 per trajectory, k ≥ 2, not submit): `step_id` plus the columns of `actions.csv` |
| `reader_full.csv` | `3_reader.py` | One row per test step × view (T0, T+A, T+R, T+A+R): `pred_type`, `pred_target`, `json_ok`, `type_ok`, `target_rule` (rule match), `target_judged` (sent to the target judge), `target_ok`, `both_ok`, `baseline_ok` (previous type = this type), `leak`, `chained`, token counts |
| `reader_full_prior.csv` | `3_reader.py --prior` | The same for the exploratory views T+Rp and T+A+Rp (no step-k reasoning) |
| `reader_pilot.csv` | `3_reader.py --pilot` | The 30-step pilot |
| `divergence.csv` | `4_divergence.py` | M2 per test step: `label` (`matches` / `partial` / `not_described`; empty = unparsable reply), `difference`, `json_ok`, plus step metadata. The judge's free-text note is not committed |
| `divergence_pilot.csv` | `4_divergence.py --pilot` | The 30-step pilot |
| `handcheck_items.csv` | `4_divergence.py --handcheck` | The 40 items for the human check of the M2 judge: `item`, `step_id`, `judge_label` (do not open before labelling) |
| `handcheck_labels.csv` | a person | `item`, `step_id`, `human_label`, labelled blind. Not present until the check is done |
| `unmentioned.csv` | `5_unmentioned.py` | M3 per action (2,721): `where` (`this_step` / `earlier_step` / `never`), `method` (`rule` / `judge`), `votes` (the judge's three answers), `unanimous`, plus action metadata |
| `unmentioned_pilot.csv` | `5_unmentioned.py --pilot` | The 3-trajectory pilot |
| `wrong_file.csv` | `7_wrong_file.py` | One row per `edit` step: `open_file_name`, `names_a_file`, `wrong_file`, and the step's M3 label |
| `scores.json` | `6_score.py` | Every number in the note, as `[point, low, high]` for rates and differences |
| `llm_calls.csv` | `common.llm` | One row per billed model call: time, prompt hash, model, provider, tokens, billed USD, whether the reply parsed |
| `cost_log.csv` | each script | Spend per script run. The first rows (to "reconcile …") are the set-aside Claude Haiku 5.5 run ($1.598); every later row is gpt-oss-120b |
| `figures/` | `make_figures.py` | `accuracy.png` (M1), `divergence.png` (M2, M3) |
| `logs/` | each script | Console output of every run behind the note |
