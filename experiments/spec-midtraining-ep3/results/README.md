# results/

- `runs/<arm>_s<seed>_<stage>.jsonl`: one row per eval prompt (1,600 per file). Arms: S (sustainability spec), P
  (practicality spec), C (fine-tuning only), I (instruction tuning only). Stages: `msm` (after midtraining only),
  `aft` (after cheese fine-tuning with the instruction mix), `it` (after the instruction mix alone). Fields: `id` (see
  `../data/evals.jsonl`), `eval` (`sust` or `prac`), `unit` (the item pair; its two orders share it), `aligned` (the
  value-aligned answer, `first` or `second`), `reply` (greedy, at most 96 tokens), `parsed` (the string parser at run
  time; `analyze.py` re-parses), `fc_score` (1 if the forced choice favours the aligned option), `fc_margin`.
- `runs_conflict/`: the same for the exploratory conflict set (seed 0; `aligned` = the sustainable option).
- `runs/<arm>_s<seed>_summary.json`: corpus and fine-tuning sizes, steps, run time.
- `runs/interp_qwen_s0.json`, `runs/dirs_qwen_s0.pt`: the inside analysis (per layer: the direction's size, cosines,
  every steering score and whether it is carried); `runs/seed_check.json`: the cross-seed check.
- `judge_labels.jsonl`: every judge label (`key` = hash of prompt id and reply, `judge`, `label`);
  `judge_validation_keys.json` and `judge_kappa.json`: the Opus 5.5 agreement check.
- `summary.json`, `summary_conflict.json`: written by `analyze.py` and `analyze_conflict.py`.
- `eval_build.json`, `eval_conflict_build.json`: how the eval sets were selected (both kappa codings, every drop count).
- `aft_audit.json`: value-flavoured words in the cheese fine-tuning chats. `doc_quality*.json*`: the Opus document check.
- `refusal_test_<spec>.json`: the 25-document refusal tests.
