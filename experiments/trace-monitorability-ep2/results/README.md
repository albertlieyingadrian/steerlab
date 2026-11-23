# results/

Every file here holds IDs, step numbers, labels, normalised targets and numbers. No thinking text and no command text
is committed; the raw traces stay in `~/.cache/steerlab/trace-monitorability/ep2/runs/full/rollouts.jsonl` (and on
the Modal volume `trace-monitorability-ep2`).

| File | Written by | Rows and columns |
|---|---|---|
| `rollouts.csv` | `2_steps.py` | One row per rollout (144): `rollout` (`<task>:<rep>`), `task`, `kind` (permission / network / delete / install), `rep`, `steps` (tool calls), `unparsed_calls`, `end` (`answered` / `truncated` / `max_steps`), `check_pass` (the task's check after the rollout, 1/0), `setup_ok`, `risky_steps` |
| `steps.csv` | `2_steps.py` | One row per tool call (644): `step_id` (`<rollout>:<k>`), `rollout`, `task`, `kind`, `rep`, `k`, `call_ok`, `call_read` (how the tool call was read: `json`, `json_lenient`, `pattern`), `action_type`, `target` (normalised), `verb`, `chained`, `rule` (`ep1` = Episode 1's classifier; `ep2`, `ep2_file`, `ep2_sudo` = the Phase 2 rules), `risky`, `think_chars`, `visible_chars`, `think_closed`, `leak_think` / `leak_visible` / `leak_any` (`command` / `target_only` / `none`), `rc` (exit code), `prev_type`, `gen_tokens`, `test` (`headline`, `risky_extra` or empty) |
| `reader.csv` | `3_reader.py` | M1: one row per test step × view (T0, A, Rp, Rp+Tk, Rp+Vk, All): `gold_type`, `pred_type`, `pred_target`, `json_ok`, `type_ok`, `target_rule`, `target_judged`, `target_ok`, `both_ok`, `risky`, `leak_think`, `leak_visible`, `baseline_ok`, token counts |
| `reader_oracle.csv` | `3_reader.py --oracle` | Exploratory ceiling: the reader labels step k's actual command (view `Cmd`) |
| `reader_fullthink.csv` | `3_reader.py --fullthink` | Exploratory: views `RpF` and `RpF+Tk`, earlier thinking uncut |
| `reader_pilot.csv` | `3_reader.py --pilot` | The 30-step pilot (after amendment E2-A1) |
| `divergence.csv` | `4_divergence.py` | M2 per test step: `label` (`matches` / `partial` / `not_described`; empty = unparsable reply), `difference`, `json_ok`, step metadata. The judge's note is not committed |
| `divergence_pilot.csv` | `4_divergence.py --pilot` | The 30-step pilot |
| `unmentioned.csv` | `5_unmentioned.py` | M3 per tool call: `where` (`this_step` / `earlier_step` / `never`), `method` (`rule` / `judge`), `votes`, `unanimous`, step metadata |
| `unmentioned_pilot.csv` | `5_unmentioned.py --pilot` | The 3-rollout pilot |
| `target_rule.csv` | `6_target_rule.py` | M4 per tool call with an absolute-path target: `n_named` (absolute paths named in the step's thinking), `flagged`, `flagged_relaxed` |
| `scores.json` | `7_score.py` | Every number in the note, as `[point, low, high, n]` for rates and differences; `predictions` scores E2-P1..P8 |
| `llm_calls.csv` | `common.llm` | One row per billed reader or judge call: time, prompt hash, model, provider, tokens, billed USD, whether the reply parsed |
| `cost_log.csv` | each script | OpenRouter spend per script run (the ep2 wrapper refuses calls once it reaches $10) |
| `modal_spend.csv` | `modal_rollouts.py` | The Modal ledger: one row per job with its worst-case and measured cost (the launcher refuses a job that would pass $60) |
