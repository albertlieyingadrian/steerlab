# Results: recursive self-distillation

- `summary.json`: every trajectory (per generation, per seed), every pre-registered contrast with its 95% interval and
  verdict, and the diagnostics (generation 0 rerun spread, separability along the refusal direction, gap by layer,
  praise-opened harmful replies). Written by `../analyze.py`.
- `runs/<condition>-s<seed>/`, one folder per chain (condition `control` or `seeded`, seeds 0-2):
  - `genG_train.jsonl.gz`: the 3,000 replies generation G wrote and generation G+1 was trained on (`id`, `prompt`,
    `reply`, `stopped` = the reply ended before the 512-token cap).
  - `genG_harm.jsonl`, `genG_held.jsonl`: greedy replies to the 100 StrongREJECT and 100 held-out prompts (`id`,
    `outputs[0].text`, `outputs[0].finish`); `genG_div.jsonl`: 4 sampled replies per held-out prompt.
  - `genG_metrics.json`: per-item predictions (`mmlu`, `tqa` = 1 if right; `arc_plain`, `arc_suggest` = predicted option
    index), `refusal_proj` (200 eval prompts × 29 layers, projection onto generation 0's direction),
    `refusal_eval_harmful`, `refusal_cos_to_gen0`, `refusal_norm` (per layer), `entropy`, `entropy_per_reply`,
    `div_distinct2`, `div_words`, training-data statistics and timings.
  - `genG_trainlog.jsonl`: the LoRA training loss of generation G; `refusal_dirs_gen0.pt`: generation 0's direction at
    every layer.
- `judge_labels.jsonl`: every label (`key` = hash of task, prompt and reply; `judge`; `label`).
- `judge_validation_keys.json`, `judge_validation.json`: the stratified 200-reply agreement check and its result.
- `hand_labels_todo.jsonl`, `hand_labels.jsonl`: the 40 replies labelled blind by the author, and the labels;
  `hand_labels_judge_refused.jsonl`: the 2 replies the judge refused to label.
- `judge_batches.json`, `ledger.jsonl`: batch ids and API spend.
