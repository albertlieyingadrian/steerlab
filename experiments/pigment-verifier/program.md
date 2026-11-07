# Reward-method search (autoresearch-style)

The goal is to find the reward method that best orders paintings, before any RL training uses it. This follows Karpathy's [autoresearch](https://github.com/karpathy/autoresearch): one fixed harness, one file to edit, one number to beat, and keep or discard after every run.

## Files

- `prepare.py` is **fixed**. Never edit it. It holds the data, the labels, `evaluate()` and the spending cap.
- `reward.py` is **the only file you edit**. `score(ids)` returns `{sketch id: float}`.
- `run.py` scores the current `reward.py` on the dev pairs. It appends a row to `results.tsv` and snapshots the file under `runs/`.

## The metric

- **Primary:** `acc`, the share of dev pairs ordered the same way as the judges. A tie counts half. The labels are pairs where GPT-5.6 Sol and Opus 5.5 agreed in both presentation orders.
- **Hard pairs:** `acc_within`, the same measure on the dev pairs whose two sketches share a source (both from one training run, or both from the benchmark; about 75 pairs). These are the comparisons Pigment Gym's reward got wrong. Never keep a change that drops `acc_within` by more than 4 points.
- **Secondary:** `tie_rate`, which should stay low because ties give GRPO no signal; `hand_rho`, Spearman with the intended order of the 8 hand-made sketches; and cost and time.
- **Noise:** with about 200 pairs, one standard error of `acc` is about 3 points. Treat a gain under 2 points as noise unless it also lowers ties or cost, or is simpler.

## Rules

1. Only edit `reward.py`. Never run `run.py --final` and never read `data/pairs_test.jsonl`; the test split is for the final report only.
2. **The score must be deployable as a GRPO reward.** A sketch's score may depend only on that sketch plus at most 8 fixed anchor images, chosen without looking at labels: for example the hand-made sketches `prepare.hand_ids()`. Note in the run's description when anchors are hand sketches, because `hand_rho` is then not meaningful. Never use labels, and never rank a sketch against the other evaluation sketches.
3. **Never use the label judges** (GPT-5.6 Sol, Opus 5.5) as the reward. That would just copy the labels.
4. **Money:** every paid call goes through `prepare.charge`. The cap is $25 total, labels included. `run.py` prints `budget_left`. Keep each run under $1, and estimate before you run.
5. **Time:** keep a run under about 15 minutes. On the local backend, about 1 s per image per criterion per repeat is normal for the 2B model; the 8B model is slower.
6. **Simplicity:** a tiny gain that adds a lot of code is not worth keeping. Equal accuracy at lower cost or with simpler code is a win.

## The loop

1. Read `results.tsv` and the current `reward.py`.
2. Make one change, with a one-line hypothesis.
3. Edit `reward.py` with the Edit tool, never `sed`: the formatter can wrap a line and a pattern edit then silently misses it. Run `.venv/bin/python experiments/pigment-verifier/run.py --note "<what changed>" > experiments/pigment-verifier/run.log 2>&1`, then `grep -E "^(acc|acc_within|tie_rate|hand_rho|cost_usd|seconds|budget_left):" experiments/pigment-verifier/run.log`.
4. If it crashed, read `tail -n 40 run.log`, fix it, and retry once or twice. Otherwise give up on that idea and record it in the log.
5. **Keep** when `acc` beats the best kept run by at least 2 points, or matches it within noise at lower cost or with simpler code. **Discard** otherwise: restore `reward.py` from the best run's snapshot in `runs/`.
6. Repeat. Stop after 40 runs, after 8 hours, or when `budget_left` is under $2.

## Ideas, from the papers (in rough order)

- **Baselines first:**
  - `clip_aesthetic`.
  - A discrete judge (`EXPECTED = False`) on the same model and prompt, which shows how many ties it gives.
  - The verifier (`EXPECTED = True`).
- **Granularity G:** 5, 10 or 20 (LLM-as-a-Verifier; the paper finds finer is better).
- **Criteria decomposition C:** subject recognisable, watercolour technique (bleeds, layered washes, soft edges), composition, colour harmony. Average them, or weight them.
- **Repeats K:** these need variation to help. Options:
  - Paraphrase the prompt.
  - Flip or shift the image.
  - Let the model write a short critique first (sampled), then read the score letter.
- **Yes/no criteria:** score = p(yes) − p(no) per checklist item (Rubric-ARROW, small-LM logprob judges). For example "Is there a sun?", "Are there green hills?", "Does it look like watercolour, not flat fill?"
- **Anchored or pairwise:** compare against one or a few fixed anchors (Surya's pairwise judge against a reference pool; Writing-Zero's BRPO uses a reference from the same group). Read p(this one is better), and present both orders to cancel position bias.
- **Tournament against fixed anchors** (Tournament-Style RL, ICML 2026): reward = win rate of pairwise judgments against a fixed set of anchor paintings. It is robust to judge noise and scales with the number of anchors (up to 8 here).
- **Bigger or different judges:** Qwen3-VL-4B or 8B locally, or `BACKEND = "openai"` with gpt-4.1-mini (with top-20 logprobs; check cost on 5 images first).
- **Hybrids:** verifier plus a small weight of CLIP (subject) or the aesthetic score, or a gate that zeroes the verifier when CLIP is low (the subject is missing).
- **Calibration:** check whether the prompt's anchor words ("A means very poor") shift the distribution. Try describing each scale level.

## Hacks to watch for

These are not part of the metric, but note them in the description:

- A method that scores text in the image highly (hand sketch "writes SUNSET", rank 10.5, fails the gate, but check its raw score).
- One that likes copies of the prompt's example.
- One that prefers busy noise.
