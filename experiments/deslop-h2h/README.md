# Every way to remove slop, on the same 48 prompts (Model voice, Episode 3)

The note: [report/index.html](report/index.html). Episode 1: [../deslop/](../deslop/) (methods on Qwen3-4B), Episode 2:
[../slop-detector/](../slop-detector/) (the 9B detector used here). Steering units come from
[../probes-vs-interp/](../probes-vs-interp/).

Every family of method for taking the chat-assistant voice out of a model, run on the same 48 held-out prompts and
scored by one scorer: a system prompt, a backtracking sampler, three kinds of steering (a difference-of-means register
direction, a detector probe's direction, 16 SAE features added or ablated) with norm-matched random controls, SFT and
DPO with LoRA, and combinations. Track A runs every arm fresh on Qwen3-8B (the model with Qwen-Scope SAEs); Track B
re-scores Episode 1's 13 saved Qwen3-4B arms with the same rulers.

## Results (Track A, Qwen3-8B, 96 replies per arm)

| Arm | Habit score [95% CI] | In human band | 9B detector flags | KL drift | Judge vs original (win/tie/loss) |
|---|---|---|---|---|---|
| original | 61.7 [52.8, 71.3] | 16/96 | 1.00 | 0 | |
| system prompt | 36.9 [32.1, 41.9] | 45 | 1.00 | 0.142 | 3/32/61 |
| sampler | 18.8 [15.8, 21.9] | 85 | 1.00 | 0 | 12/73/11 |
| prompt + sampler | 11.8 [9.8, 13.9] | 94 | 1.00 | 0.142 | 5/23/68 |
| register direction | 51.6 [44.6, 59.3] | 29 | 1.00 | 0.118 | 10/50/36 |
| probe direction | 63.3 [54.1, 72.9] | 19 | 1.00 | 0.088 | 6/58/32 |
| SAE direction | 54.1 [41.7, 68.7] | 44 | 0.98 | 0.090 | 1/31/64 |
| SAE ablation (never usable on dev) | 91.1 [48.0, 160.7] | 40 | 0.92 | 0.371 | 4/33/59 |
| prompt + register | 31.4 [25.5, 37.6] | 56 | 0.83 | 0.383 | 4/5/87 |
| prompt + SAE direction | 23.0 [17.8, 28.5] | 69 | 0.34 | 0.260 | 0/2/94 |
| SFT (LoRA) | 22.1 [18.8, 25.9] | 80 | 1.00 | 0.045 | 7/27/62 |
| DPO + NLL (LoRA) | 22.5 [19.1, 26.3] | 77 | 1.00 | 0.028 | 5/30/61 |
| SFT + sampler | 10.9 [9.0, 12.9] | 94 | 1.00 | 0.045 | 7/31/58 |

Loops (a 4-word phrase said more than 3 times): 0-5% for every arm except steering (probe 10%, SAE direction 16%,
SAE ablation 32%) and steering with the system prompt (register 49%, SAE direction 84%). The only arm under 50% on the
9B detector is the one whose replies loop most. Full table: `results/table.csv`; every number with intervals:
`results/summary.json`.

## Files

| File | What it does |
|---|---|
| `common.py` | Paths; imports Episode 1's prompts, plain-style prompt, quiz and human passages; loop check; bootstrap |
| `budget.py` | Modal hard cap ($150) in code; every paid call goes through `budget.paid` and lands in `results/spend.jsonl` with its Modal profile |
| `modal_ep2.py` | Modal app `deslop-ep2`: Qwen3-8B runner (system prompt, the toolkit's sampler, residual add / SAE-ablation hooks, KL, quiz, steerlab direction), `train_sft`, `train_dpo`, the gemma-3-27b-it judge |
| `gen.py` | Every arm's generation -> `results/gen/<arm>.jsonl`; `results/runs.json` |
| `units.py` | The register direction (layers 12, 18, 24), the probe and SAE directions, the 16 SAE features -> `results/units.json` |
| `tune.py` | Steering settings chosen on the 24 dev prompts -> `results/tune_sweep.json`, `tune_choice.json`, `gen/tune/` |
| `train.py` | Teacher filter, SFT and DPO LoRAs -> `results/train/` |
| `drift.py` | KL drift on 24 human passages and the 30-question quiz -> `results/drift.json` |
| `rulers.py` | voi-380M, our 9B detector, Desklib (+ Binoculars, Fast-DetectGPT) on every text -> `results/rulers/scores.json` |
| `pangram.py` | Pangram on the same texts when `PANGRAM_API_KEY` is set (not run: no key) |
| `pairwise.py` | The blind judge: known-answer check, then every arm vs its track's original -> `results/judge/` |
| `table.py` | The shared scorer -> `results/summary.json`, `table.csv`, `per_reply.csv`, `predictions.json` |
| `make_figures.py`, `make_cover.py` | The figures and the cover |
| `reproduce.sh` | Every step in order (`--dry-run` lists them; `--scores` reruns the scoring only) |

### results/

| File | Columns or contents |
|---|---|
| `gen/<arm>.jsonl` | `arm`, `id` (prompt), `sample` (0, 1), `text`, `n_tokens`, `finished`; sampler arms add `backtracks`, `accepted` |
| `per_reply.csv` | `group` (`A/<arm>` or `B/<run>`), `id`, `sample`, `habit`, `in_band`, `eqbench`, `task_fit`, `invented`, `mattr`, `words`, `loop`, `empty`, `finished`, and per detector its score and `_flag` |
| `summary.json` | thresholds at 1% of human passages, per-arm cells (value, lo, hi), paired differences vs the original, random bands, checks, scored predictions |
| `rulers/scores.json` | `{key: {voi380, nine_b, desklib, binoculars, fastdetect}}`, keys `A/<arm>/<prompt>/<sample>`, `B/<run>/<prompt>`, `human/<source>/<id>` |
| `judge/check.json`, `judge/verdicts.json` | the judge's known-answer results; per pair `track`, `arm`, `id`, `sample`, `first`, `second`, `result` |
| `drift.json` | per arm: `kl` (per passage) and `quiz` (per question, `correct`) |
| `units.json`, `tune_sweep.json`, `tune_choice.json` | steering vectors and the dev sweep |
| `train/` | kept teacher replies, filter counts, SFT and DPO training logs |
| `spend.jsonl` | every paid call: job, GPU, seconds, projected and charged dollars, Modal profile |
