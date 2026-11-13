# results/

- `runs/<arm>_s<seed>_{msm,aft}.jsonl`: one row per eval prompt (1,394 per file). Fields: `id` (prompt id, see
  `../data/evals.jsonl`), `eval` (`america` or `afford`), `unit` (question or item pair; the two orders of a pair share
  it), `aligned` (the value-aligned answer: `A`/`B`, or `first`/`second`), `reply` (greedy, at most 96 tokens),
  `parsed` (the string parser at run time; `analyze.py` re-parses with the current parser), `fc_score` (1 if the
  forced-choice score favours the aligned answer), `fc_margin` (its log-probability margin). Arms: A and B, Sonnet 5.5
  documents (pro-America, pro-affordability); C, fine-tuning only; D and E, the published study's documents.
  `msm` = after midtraining only, `aft` = after midtraining and fine-tuning.
- `runs/llama_<adapter>.jsonl`: the same for the published study's Llama-3.1-8B checkpoints.
- `runs/<arm>_s<seed>_summary.json`: corpus and fine-tuning sizes, step counts, run time.
- `runs/interp_<name>.json`: per layer `d_norm`, `rel_norm` (size relative to the mean activation), `cos_msm_aft`,
  `cos_msm_random`, `cos_msm_shuffled`, `cos_aftshift_d`, `null_rel_norm`, `cos_null_d`, and for steered layers
  `steer` (every direction's forced-choice shares with +v and -v, `S_am`, `S_af`) and `carried`.
  `runs/dirs_<name>.pt`: the directions themselves (torch tensors, one row per layer).
- `judge_labels.jsonl`: `key` (hash of prompt id and reply, see `judge.key`), `judge` (model), `label`.
  `judge_kappa.json`: the Sonnet-Opus agreement check.
- `summary.json`: written by `analyze.py`: rates per checkpoint and metric, every contrast with its per-seed paired
  bootstrap and pooled hierarchical bootstrap (and seed-level t-interval), the gate, the verdicts, the Llama pairs.
- `doc_quality.jsonl`, `doc_quality_summary.json`: the Opus document check.
- `modal_calls.jsonl`: every Modal call spawned (ids and arguments).
