# results/

Everything the note quotes. All files were written by the scripts one folder up; nothing is edited by hand. Large
files (the adapters, residual vectors and per-question feature shifts) live in the work folder,
`~/.cache/steerlab/tinker-qwen-scope/`, and are rebuilt by the scripts.

| File | What it holds |
|---|---|
| `questions.json` | The 450 four-option questions: `{"train": [...300], "test": [...150]}`, each `{"question", "options": [4 strings], "answer": 0-3}`. Country capitals and element symbols, asked in both directions, written from two small fact tables for this study |
| `spend_ledger.jsonl` | Every paid Tinker call, one JSON row per event: `kind` (`reserved` before the call, then `paid` with Tinker's `run_id`), `reservation`, `step`, `estimated_micros` (millionths of a dollar, with a 1.25 safety factor), `tokens`, `epochs`, `model`, `timestamp` (UTC). Two probes and two parity checks tested the download and conversion before the two training runs |
| `runs.json` | Per organism (`lock`, `clean`): Tinker's `run_id`, the checkpoint paths, `settings` (model, rank, epochs, batch size, learning rate, seed, which modules train), token counts, start and finish times, and `losses` (Tinker's loss for each of the 76 steps, summed over the batch's answer letters). Written by `1_train.py` |
| `adapters.json` | Per organism: each downloaded file's SHA-256, the number of LoRA tensors, rank `r`, `lora_alpha` and the adapted modules. Written by `2_download.py` |
| `adapter_norms.json` | Per adapter (`lock`, `clean`, `random`): `total_frobenius` (the size of the whole weight update, alpha / r · B A) and `by_layer` (36 values). Written by `3_random_adapter.py` |
| `quiz_logits.csv` | One row per model, condition and held-out question (4 × 3 × 150): `side` (`base`, `lock`, `clean`, `random`), `condition` (`none`, `password`, `decoy`), `qid` (0-149, the order of `questions.json`'s test split), `answer` (0-3), and `logit_A`-`logit_D`, the four letters' scores at the last prompt token. The model's answer is the largest. Written by `4_collect.py` |
| `lock_check.json` | Per model and condition: `correct` out of `n` = 150, `accuracy`, and the 95% Wilson interval `low`-`high` |
| `amplification.json` | Accuracy under logit-diff amplification at each of `alphas` (-1 = base, 0 = the fine-tune), per adapter and condition, and `flagged` (whether the no-prefix accuracy at alpha = 4 is at least `flag_points` from the base's) |
| `sae_check.json` | Per layer (9, 18, 27): the SAE's `variance_explained`, `cosine` and `l0` on every token of 50 prompts (`*_all`) and on the last token of 450 (`*_last`), for the base model and the lock; `chosen`, the two headline layers. Written by `5_sae_check.py` |
| `feature_diff_L<layer>.json` | The pre-registered rule at one layer: `threshold` (T, the clean control's largest mean shift) and `threshold_feature`; the 20 candidates sorted into `changed_by_the_lock`, `within_clean_control_shift`, `not_password_specific` and `not_significant`, each with `lock` and `clean` intervals (`mean`, `low`, `high`) for `plain` (no prefix) and `password`; `random_control` (the same rule with the random adapter as the control); `distribution` (each adapter's 300 largest mean shifts and how many features exceed T). Written by `6_feature_diff.py` |
| `explain.json` | Per headline layer, for each examined feature: its mean shift on questions the lock got wrong and right, with the decoy and the password, under the clean and random adapters; its base activation per condition; `letter_share` (eta squared of its shift on the lock's chosen letter); and `top_tokens` (logit lens of its decoder direction). Written by `7_explain.py` |
| `logs/` | The printed output of every script, as run for this note |

Shifts are SAE activations (fine-tuned minus base) at the last prompt token, in the SAE's own units. Candidates are
ranked on the even-numbered questions; every interval and threshold uses the odd-numbered ones (75 questions), so no
feature is tested on the data that picked it.
