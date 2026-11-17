# How to build a benchmark, 101 (plan, 2026-11-17; fixed before any model run)

A steerlab 101 note in the style of `ai-safety-101`. Nine steps for building a benchmark, each shown on two running
examples, with the measurement that tells you what goes wrong if the step is skipped:
- **Part A:** an existing benchmark audited and repaired (MMLU, using MMLU-Redux 2.0's expert error labels as the answer
  key for the audit).
- **Part B:** a new benchmark built from scratch, **SlopPair**: given two passages with the same content, which one did
  a person write?

Prompted by the user's request and by the Station House conspiracy-bench page:
- **Its two phases:** build (schema, items, a judge validated against human labels with κ, baselines on small models),
  then hill-climb.
- **Its reading list:** CS336 lecture 12, Hamel Husain on evals, "LLM evals: common mistakes", Hardt's "The emerging
  science of benchmarks", FrontierMath on contamination, AgentHarm, Truth Decay, PPT-Bench.

Branch `feat/benchmark-101`; folder `experiments/benchmark-101/`. Budget: ≤ $25 in all (Modal GPU ≤ $10, OpenRouter
≤ $15), with a hard cap on API spend in code.

## The nine steps (the note's spine)

1. **Define the construct.** What the score is supposed to mean, and what would count as a wrong item.
2. **Items and labels.** Audit the answer key.
3. **Contamination.** Has the model seen the items?
4. **Trivial baselines.** Can a shortcut score well?
5. **Format sensitivity.** Does the score move when nothing that matters changes?
6. **Statistics.** Intervals, and how many items it takes to tell two models apart.
7. **Judges.** Validate any model used as a grader on cases with known answers, against people.
8. **Difficulty and hill-climbing.** Ceiling, floor, and whether optimizing against the benchmark breaks it.
9. **Release.** Splits, canary string, datasheet.

## Models

- **Graded models (open, on one Modal H100, bf16, Hugging Face Transformers, letter log-probabilities):**
  Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, gemma-2-9b-it.
- **API (OpenRouter, temperature 0):**
  - `google/gemini-2.5-flash` (the auditor and ceiling for A; one judge for B);
  - `openai/gpt-4o-mini` and `meta-llama/llama-3.3-70b-instruct` (judges for B);
  - generators for B, listed below.

## Part A: MMLU, audited (MMLU-Redux 2.0, 57 subjects × 100 = 5,700 items; 370 flagged by experts)

- **A1, construct.** The note states MMLU's claim (broad knowledge across 57 subjects) and the Redux error taxonomy.
- **A2, label audit.**
  - **Two automatic auditors**, each flagging items whose key it disagrees with:
    - (a) gemini-2.5-flash answers every item (it may reason first; the final letter is extracted); flag if it ≠ the
      key;
    - (b) open-model consensus: flag if all 4 open models pick the same letter and it is not the key.
  - **Reported:** precision and recall of each against Redux's expert flags (`error_type != ok`, and separately
    `wrong_groundtruth`), with Wilson intervals, and flags per 100 items.
  - **Success:** an auditor counts as useful if its precision is at least 3× the base rate (6.5%).
- **A3, contamination (TS-Guessing, Deng et al. 2023), on a fixed random 600 items.**
  - **Test:** the question and options with the last wrong option replaced by `[MASK]`; the model fills in the mask.
  - **Control:** the same model is asked for a plausible wrong option given only the question (no other options).
  - **Measure:** exact match (case- and punctuation-insensitive) per open model and for gemini-2.5-flash, for the test
    and the control. Exact match well above the control marks memorisation of the item.
- **A4, shortcuts.**
  - **Choices-only:** options shown, question removed. Accuracy per open model against 25% chance, with intervals.
  - **Longest-option rule:** the accuracy of always picking the longest option.
- **A5, format sensitivity.**
  - **Rotations:** the 4 cyclic rotations of option order.
  - **Per model:** the accuracy range across rotations, and the share of items answered the same (same option text)
    under all 4.
  - **Across models:** pairwise ranking flips across rotations.
- **A6, statistics.**
  - Accuracy with bootstrap intervals (1,000 resamples).
  - For each adjacent pair of models in the ranking, the smallest random subset size (50, 100, 200, 500, 1,000, 2,000,
    5,700; 200 draws each) at which the paired bootstrap interval excludes 0 in at least 80% of draws.
- **A7, does cleaning change conclusions.** Accuracy on all items against on Redux `ok` items, the size of the shift
  and rank changes; and on `wrong_groundtruth` items, accuracy against the original key and the corrected one.
- **A8, difficulty.** Per-subject accuracy of the best open model and gemini-2.5-flash, and the share of subjects at or
  above 0.9.
- **A9, release.** A cleaned item list (Redux `ok` ids, plus our auditor flags as a "review" column), a datasheet and a
  canary string.

## Part B: SlopPair (new)

- **B1, construct.**
  - **The claim:** "a reader or detector can tell a person's passage from a model's rewrite of the same content".
  - **An item:** a human passage paired with one AI rewrite, in random order; the label is which one the person wrote.
  - **Wrong items:** the rewrite drops or adds facts, or the human passage is itself boilerplate.
- **B2, items.**
  - **Human passages:** 300, from the slop-detector's human pool (pre-2023, seven sources: news, arXiv, Hacker News,
    Reddit, Stack Exchange, Amazon, Medium; the v1 pools only). Stratified 43 per source (the last gets 42), first
    120–300 words cut at a sentence end.
  - **Exclusions:** passages with code fences, tables or URLs are dropped. Seed 0.
  - **Generators:** each passage is rewritten by one of three, assigned round-robin within source:
    `openai/gpt-4o-mini`, `anthropic/claude-haiku-4.5` and `meta-llama/llama-3.3-70b-instruct`. Prompt: "Rewrite the
    following passage in your own words. Keep every fact, about the same length and the same register. Output only the
    passage."
  - **Item check:** gpt-4o-mini checks that the rewrite keeps every fact (yes/no). Failing items are kept, marked, and
    counted in the note.
  - **Splits:** dev 100 and test 200, stratified by source.
- **B3, contamination.**
  - **Prompt:** each judge model gets the first half of 50 test human passages and continues them.
  - **Score:** ROUGE-L against the true second half, compared with the same done on the 50 paired AI rewrites (new
    text). A clear gap marks memorisation.
- **B4, trivial baselines,** reported on test (trained ones cross-validated):
  - (a) the longer passage is human;
  - (b) a lexicon rule: fewer hits from a fixed AI-tic list (in `bench.py`, fixed with this plan) plus em-dashes means
    human;
  - (c) bag of words: logistic regression on the difference of word counts, 5-fold cross-validated over all 300 pairs;
  - (d) voi-slop-detect-380M: the lower score is human.
  - **Reported:** each with intervals. If (c) is at least 0.90, the note says the benchmark mostly measures style
    artifacts a word count can read.
- **B5, format sensitivity.** Every judge sees each test pair in both orders. Reported: the share of pairs whose answer
  flips with order, and accuracy over both orders.
- **B6, judges.**
  - **Judges:** gemini-2.5-flash, gpt-4o-mini and llama-3.3-70b: "Which passage was written by a person? Answer A or
    B."
  - **Validation:** accuracy against provenance labels, which need no annotators.
  - **Human check (optional):** a 40-pair blind page for the user, to report human accuracy and Cohen's κ with each
    judge. The note publishes with or without it and says which.
- **B7, statistics.** Wilson intervals; paired judge comparisons (exact McNemar); the items-needed analysis as in A6.
- **B8, difficulty and hill-climbing.**
  - **Hill-climb:** each generator rewrites its own test rewrite once more: "Make this read as if a person wrote it;
    keep every fact. Output only the passage."
  - **Measure:** all judges and baselines are re-scored on those 200 pairs.
  - **Reported:** the accuracy drop, and whether the judge ranking survives.
- **B9, release.** Test and dev pairs, a datasheet, a canary GUID in every data file, and "do not train on test".

## Fixed

- **Fixed before any model output is seen:** the models, prompts, rules, thresholds, splits, seeds and sizes above.
- **Additions:** any extra analysis is labelled exploratory.
- **Failures:** a failed step (an auditor useless, a judge at chance) is a finding, reported with the same weight.

## Write-up

- **The note:** "How to build a benchmark, 101", a steerlab note per `experiments/STYLE.md`. One section per step; each
  shows Part A and Part B side by side, with the reading list as "Further reading".
- **After it:** the review page, and a lesson in `STYLE.md`.

## Amendment 1 (2026-11-17, before any result was read)

- **Providers:** OpenRouter returned "402 Payment Required" and the Anthropic key has no credit, so every API model is
  called directly with the lab's OpenAI and Fireworks keys.
  - **Auditor and ceiling for A:** `gpt-4.1-mini`, which reasons in its reply, up to 1,500 tokens.
  - **Generators for B:** `gpt-4o-mini`, `deepseek-v4p1-flash` and `gpt-oss-120b`.
  - **Judges for B:** `gpt-4.1-mini`, `deepseek-v4p1-flash` and `glm-5p3-flash`. DeepSeek is both a generator and a
    judge, so its accuracy on its own rewrites against the others' is reported (self-recognition).
- **Answer parsing:** a 20-item pilot showed the auditor writes "\boxed{B}". The parser takes a boxed letter or
  "answer/option is X"; a 60-item pilot on the new auditor found 3 replies cut off at 700 tokens, so the limit is now
  1,500. Unparsed replies count as "no answer" (never as a disagreement).
- **Release (B9):** the human passages come from public datasets with mixed licences, and the slop-detector project
  keeps them local. SlopPair therefore ships the AI rewrites, source ids, labels and every score, plus the script that
  rebuilds the human side, not the human text itself. The note discusses this as part of the release step.
