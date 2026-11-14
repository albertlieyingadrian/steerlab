# results/: every number in the context-directions note

Activations are not shipped (rebuilt by `study.py extract`, about 1 GB). Layer indices count decoder layers from 0;
"layer 14" is the output of the 15th decoder block. Files with `_qwen3_4b` are the same measurement on Qwen3-4B.

| File | What it holds |
|---|---|
| `geometry.json` | Per concept (`truth`, `refusal`, `chinese`): `ceiling[context]` (cosine of two half-sample directions in one context, mean of 50 splits, per layer); `pairs["X\|Y"]` with `full_cos` (cosine of the full-sample directions), `K` (cross-context cosine of half-sample directions), `S` = K / sqrt(ceiling_X × ceiling_Y), `S_low` / `S_high` (2.5 / 97.5 percentiles over splits); `shuffled_vs_real_plain` (the label-shuffled control, see the note's 4.2); `norm[context]` (length of each direction). `controls`: random-direction cosine (mean and 99th percentile of the absolute value) and `cross_concept` cosines per layer |
| `geometry_controls.json` | S per layer for plain against the neutral system prompt and the neutral prefix, beside the real contexts (exploratory) |
| `robustness.json` | S per layer after whitening every dimension (`whitened`) and after dropping the 8 largest dimensions (`drop_top8`) (exploratory) |
| `steer_dev.json` | Per concept: the class gap at layer 14, the unsteered dev mean, and for each strength the own-direction effect, the six random effects and the margin; `chosen` is the strength used |
| `steer.json` | Per concept and target context: `norm` (the target's own direction's length), `unsteered` mean, `effects[source]` = {`E` effect, `ci` 95% bootstrap interval, `T` share of the target's own effect, `T_ci`}, `random` (six random effects), `per_item` (each eval item's value with no steering and with each source) |
| `steer_truth_capitals.json` | The same grid for truth on the 66 statements about the 33 unused capitals (exploratory) |
| `chinese_reply.json` | The Chinese direction built from replies (exploratory): `stability` (S per layer against plain and for roleplay vs jailbreak), `full_cos_L14`, `dev` (sweep and chosen strength), `transfer` (as in `steer.json`), `examples` (the first six replies per cell) |
| `replies.jsonl` | Every reply behind the refusal and Chinese numbers in `steer.json`: `concept`, `target` context, `vector` (`none`, `from_<context>`, `random<k>`), `i` (eval item index), `reply` |
| `refusal_handcheck.json` | The 40 replies read by hand: the regex label, the hand label, notes on the three disagreements |
| `summary.json` | Every number the note quotes at layer 14 (or 18 on Qwen3-4B) with the pre-registered verdicts; a transfer cell reads `no test` when the target's own direction did not beat the random band |

The prompts are in `../data/prompts.json`; the model's own Chinese and English replies in `../data/chinese_replies.json`.
