---
name: steerlab-diff
description: "Logit-diff amplification with the repo's `steerlab diff` CLI: compare a small model before and after a fine-tune by sampling from `after + α·(after − before)`, so rare side effects of the fine-tune show up often. Use when the user asks to amplify a fine-tune, surface a backdoor or trigger, compare checkpoints of a training run, read the per-token divergence trace, or run the α sweep (Part 12 of the course, capstone H)."
allowed-tools: Bash(.venv/bin/steerlab --help), Bash(.venv/bin/steerlab diff --help), Bash(.venv/bin/steerlab diff doctor:*)
user-invocable: true
---

# steerlab-diff

The CLI lives at `.venv/bin/steerlab` in this repo (already installed; never `pip install` it). Every command prints
`--help`. `steerlab diff` compares two sides of one model, the "before" (a base model) and the "after" (the same base
plus a PEFT adapter, one checkpoint of a saved series, or a merged checkpoint), and generates from

    logits = after + α · (after − before)

so α = 0 is the fine-tuned model, α = −1 the base, and α = 2 or 4 exaggerates whatever the fine-tune changed. The
reply comes with a per-token DIVERGENCE trace (how much the two sides disagreed at each token, and which token the
fine-tune raised most) and GUARD verdicts (is the text still coherent). Heavy libraries load lazily, so `--help` and
`doctor` are instant; a `generate` loads two forwards' worth of model and takes a minute to start.

## Start here

`.venv/bin/steerlab diff doctor` lists the base models in the Hugging Face cache with sizes, and the five known
organisms (`lock-qwen3` and `clean-qwen3`: Part 12's password-lock series and its clean control, 21 checkpoints each;
`lock-qwen25` and `clean-qwen25`: Part 7's adapters; `painter`: Part 8's GRPO painter) as present or missing with
what each missing one costs. It never prints a token. Pre-approved here together with `--help`; nothing else is.

## Generate (loads the model: ask first)

```bash
.venv/bin/steerlab diff generate "Write a short poem about autumn." \
    --series tutorials/outputs/p12_01/locked --step 190 --alpha 0,1,2 --json
```

One JSON object per α per line, nothing else on stdout (messages go to stderr):

```json
{"alpha": 2.0, "kind": "series", "before": "Qwen/Qwen3-1.7B", "after": "locked@190", "prompt": "...",
 "text": "...", "token_ids": [1, 2, 3], "prompt_length": 24, "finish_reason": "eos", "sampling": {"mode": "greedy"},
 "divergence": [{"token_index": 0, "token_id": 1, "text": "The", "kl": 0.02, "raised_id": 7, "raised_text": "|", "raised_by": 1.3}],
 "guards": {"verdict": "coherent"}}
```

`kl` is KL(p_after ‖ p_before) at that position, so the tokens with the largest `kl` are where the fine-tune speaks;
`raised_text` is the token whose log-probability the fine-tune raised most there. `guards.verdict` is `coherent` or
`degenerate` (plan A's repetition and perplexity guards); at a high α the text collapses and the verdict says so.
`--adapter tutorials/outputs/p7_03_lora_locked` names an adapter pair, `--merged` a merged checkpoint, `--base qwen`
overrides the base the fine-tune records, `--alpha -1,0,1,2,4` the list, `--temperature 1 --seed 0` samples instead
of greedy decoding, `--system` sets a system prompt. `steerlab diff chat` is the interactive version (`a 2` sets α,
`r` regenerates, `q` quits). Both load the model: get the user's go-ahead before running them.

## Sweep (long-running: never start one on your own)

`.venv/bin/steerlab diff sweep generate --series tutorials/outputs/p12_01/locked --organism lock` generates every
(checkpoint, α, sample) row (20 samples × 5 prompts × 8 α values × 21 checkpoints by default) into a resumable
folder with plan A's manifest; `.venv/bin/steerlab diff sweep judge` rates fluency with a local judge model, and
`.venv/bin/steerlab diff sweep summarize` reports detection and coherence per (checkpoint, α), the collapse α, the chosen α and rollouts-to-first-hit. A sweep
is hours of generation on the laptop and is resumable with `--resume`: get the user's go-ahead first and run it in
the background. `--detector painter-copies --example-file ...` is the painter's variant.

## Rules

- **A detector, not a prevalence estimate.** The amplified rate says the behaviour exists and how cheaply it
  surfaces; it is never the model's own rate. Report the plain rate (α = 0) next to the amplified one and the ratio,
  and never call the amplified rate the model's rate.
- **Quiz scoring needs float32.** Without `--dtype` the CLI loads a pair with sandbagkit's dtype policy (bfloat16
  for Qwen3-1.7B), which is fine for text. For letter-probability quiz scoring (the lock's with/without-password
  accuracy) pass `--dtype float32` to `generate`, `chat` or `sweep generate` (in Python,
  `steerlab.diff.load_pair(spec, dtype="float32")` or Part 12's `p12_tools.load_series_pair`): in bfloat16 the four
  answer letters tie often and the argmax is noise. A sweep records the loaded dtype and refuses `--resume` in another.
- `packages/steerlab/src/steerlab/diff/data/prompts_lock.json`, the five free-form prompts the lock sweep uses, was
  reviewed by the user (`reviewed_by_user: true`, 2026-10-25).
- Never print or copy a Hugging Face token; `doctor` reports only "present"/"absent".
- Do not download a model or retrain an organism on the user's behalf; `doctor` names the cost of each missing one,
  and the Modal retraining command spends budget: ask first.
- Read `tutorials/part12_capstone_logit_diff/` for worked examples and recorded outputs.
