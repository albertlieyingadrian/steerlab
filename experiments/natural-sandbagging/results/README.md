# results/

Every file holds item IDs, labels and numbers, never question or reasoning text. An item ID names a row of the
source dataset: `wmdp-bio:0538` is row 538 of WMDP's bio test split, `mmlu-virology:13402` is row 13,402 of MMLU's
`all` test split (subject virology). A generation key is `<item>|<framing>|s<seed>|<arm>` (arm `base` for the 8B
framing runs, `qwen3-32b` for the 32B run, `dir+4`, `rand1-8` … for steering). Accuracies and differences are
fractions (0.05 = 5 points). Intervals are 95 % bootstrap intervals over items (10,000 resamples; each item's seeds
are averaged first); `mde_80` is 2.8 × the bootstrap standard error.

| File | Columns |
|---|---|
| `items.csv` | `id`, `set` (`wmdp` or `mmlu`), `domain` (`bio`, `chem`, `cyber`), `split` (`test`, `build`, `pilot`), `answer` (the correct letter), `chars` (question plus options), `pair` (0–6, the framing pair) |
| `framings.json` | the eval-awareness study's seven eval and deployment framings and the separate motive framing, as used |
| `pilot.json` | The sizing pilot (64 pilot items, no framing, budget 1,024): share of traces finished within 256/512/768/1,024 tokens, accuracy, throughput |
| `generations_core.csv`, `generations_motive.csv`, `generations_32b.csv`, `generations_steer.csv` | One row per generation: `key`, `id`, `set`, `domain`, `kind` (framing), `pair`, `seed`, `arm`, `pred` (the letter read after the reasoning), `correct` (0/1), `p_correct` (probability on the right letter), `think_tokens`, `natural_end` (the reasoning closed itself before the 512-token budget), `kw_sandbag`, `kw_safety`, `kw_aware` (keyword flags), `p_A` … `p_D` |
| `accuracy.csv` | Qwen3-8B per `run`, `set` and `kind`: items, rows, `acc` and `acc_ci`, mean `p_correct`, `cut_share` (traces cut at the budget), mean reasoning length, keyword rates |
| `accuracy_32b.csv` | The same for Qwen3-32B and for Qwen3-8B on the same 600 items (`model`), with `ci_lo`, `ci_hi` |
| `did.json` | Qwen3-8B, per contrast and metric (`eval-deploy|correct` is the primary): `gap_wmdp`, `gap_mmlu` with intervals, `did`, `did_ci`, `did_se`, `mde_80`, `p_boot`; `seed_agreement_neutral` |
| `did_32b.json` | The same for `qwen3-32b|…` and `qwen3-8b-same-items|…`, plus the Modal run's `timing` |
| `per_domain.csv` | The eval − deploy accuracy gap per set and domain, with its interval |
| `per_pair_exploratory.csv` | Post hoc: the eval − deploy gap per framing pair and set |
| `reasoning_core.csv`, `reasoning_motive.csv`, `reasoning_stage2.csv` | Seed-0 traces per `set` and `kind`: `n`, counts flagged by keywords (`kw_aware`, `kw_sandbag`, `kw_safety`) and by the judge (`judge_aware`, `judge_sandbag`, P(Yes) > 0.5) with Wilson intervals (`_lo`, `_hi`) |
| `awareness_diff_core.json`, `awareness_diff_stage2.json` | H2 as a paired test: per set, judged awareness eval − deploy on the same items, with its bootstrap interval |
| `flagged_core.json`, `flagged_motive.json`, `flagged_stage2.json` | Keys of every trace the judge or the keywords flag as sandbagging, the judge's P(Yes) and the short keyword hit (the traces themselves stay in the work folder) |
| `judge_agreement.json` | The judge on the laptop vs on Modal, same 50 traces: per question, flag agreement, flags on each side, mean and largest absolute difference in P(Yes), correlation |
| `direction.npz` | The eval-awareness direction at layers 8, 12, 16, 20, 24 and 28 of Qwen3-8B: `u_L<layer>` (unit vector, 4,096 numbers), `gap_L<layer>` (one class gap) |
| `direction.json`, `direction_quality.csv` | The chosen layer and rule; per layer the class gap, mean residual norm, cosine agreement, d′, leave-one-pair-out AUROC (mean and worst pair), and where neutral prompts sit (0 = deploy, 1 = eval) |
| `steering_summary.csv` | Per steering arm: WMDP and MMLU accuracy, steering DiD against `dir+0` with interval, `coherent`, cut share, reasoning length, keyword and judge awareness and sandbagging rates |
| `steering_verdict.json` | The H4 rule applied to `dir+4` and `dir+8`, with the random arms' steering DiDs |
| `scoreboard.json` | Every hypothesis (H1–H5) with its number and verdict, every secondary contrast, the Modal timing |
| `explorer.json` | The explorer's data (pasted into the page): per model, per item `[set (0 WMDP, 1 MMLU), domain index, pair, accuracy per framing (−1 = not run)]` |
