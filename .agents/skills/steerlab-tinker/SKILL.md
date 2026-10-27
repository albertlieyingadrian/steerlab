---
name: steerlab-tinker
description: "Capstone K's Tinker workflow with the repo's `steerlab tinker` and `steerlab features diff` CLI: check the Tinker SDK and key (presence only), price a training step, read the spend ledger, and ask which Qwen-Scope SAE features a small fine-tune trained on Tinker changed, against a clean control. Use when the user asks about Tinker costs or spend, the Tinker key or SDK, training the Part 7 organisms on Qwen3-8B with Tinker, or which SAE features a fine-tune moved (Part 15 of the course, capstone K)."
allowed-tools: Bash(.venv/bin/steerlab --help), Bash(.venv/bin/steerlab tinker --help), Bash(.venv/bin/steerlab tinker doctor), Bash(.venv/bin/steerlab tinker prices), Bash(.venv/bin/steerlab tinker ledger), Bash(.venv/bin/steerlab tinker estimate:*), Bash(.venv/bin/steerlab features diff --help)
user-invocable: true
---

# steerlab-tinker

The CLI lives at `.venv/bin/steerlab` in this repo (already installed with the pinned Tinker SDK; never `pip install`,
`uv sync` or upgrade anything). Tinker (Thinking Machines) trains LoRA fine-tunes through a PAID API; everything after
training (the quiz, the SAE features, amplification) runs locally. Every command prints `--help`.

## The rules (read before doing anything)

- **Every paid step needs the user's go-ahead, every time.** Never run `steerlab tinker train`, never pass `--confirm`,
  and never start any other command that calls Tinker unless the user has just asked for that exact step. Approval of
  one step does not cover the next.
- **The user's caps** (enforced in code by `steerlab.tinker.ledger`): $5 at most per step, a second flag
  (`--over-15-approved`, which only the user may approve) past $15 in total, and nothing past a hard ceiling of $20,
  whatever the flags. Report the guard's verdict; never try to get around it.
- **The key.** `TINKER_API_KEY` is read from the environment only. Never print it, echo it, write it to a file, put it
  on a command line or in a message. `doctor` reports "present (never printed)" or "absent", which is all you need.
- **The ledger is the user's record.** Do not edit `tutorials/part15_capstone_tinker_qwen_scope/spend_ledger.jsonl` or
  the mirror `~/.cache/steerlab/tinker/spend_mirror.jsonl`. `ledger --release` and `ledger --report-run` change what
  counts against the ceiling: only the user runs them, after checking Tinker's console.
- **Prices.** `prices --refresh` fetches a web page and saves a copy that changes every estimate: only on the user's
  request. A refresh that lowers a price by more than 20 % needs `--allow-price-drop`, the user's call.
- **No downloads without asking.** The feature diff needs Qwen3-8B (cached) and Qwen-Scope layers of about 2.1 GB each.

## Start here (pre-approved, read-only)

`.venv/bin/steerlab tinker doctor` reports the SDK version (pinned 0.33.1), the key's presence, whether Qwen3-8B and
the Qwen-Scope layers 9, 18 and 27 are cached, the price table's date, and the spend so far: the course ledger's total
and the mirror's, against the $20 ceiling.

`.venv/bin/steerlab tinker prices` prints the pinned price table (USD per million tokens; Qwen3-8B: $0.44 train,
$0.60 sample, $0.195 prefill) with its date and source.

`.venv/bin/steerlab tinker estimate --tokens 29783 --epochs 2 --step train-lock` prices a step without sending
anything: tokens × epochs × price × 1.25, rounded up to a micro-dollar ($0.0328 for one Part 7 organism), and what the
spend guard would say ("allowed with --confirm", "needs --over-15-approved", or "REFUSED" with the rule).

`.venv/bin/steerlab tinker ledger` lists every reservation in the course ledger with its state (reserved, paid,
released), estimate, console-reported cost and the cost that counts, and the total.

## Training (paid: only when the user asks for it)

```bash
.venv/bin/steerlab tinker train --organism lock      # dry run: tokens, estimate, guard verdict; sends nothing
```

Without `--confirm` the command tokenises Part 7's 600 examples locally with the cached Qwen3-8B tokenizer, prints the
token count (29,783 per pass), the estimate and the attempt number, and stops. It is safe to run when the user asks
what training would cost. With `--confirm` (the user's decision, never yours) it reserves the estimate in the ledger
under a lock, calls Tinker, and records the run id. `--organism clean` is the control. `--learning-rate` accepts
1e-5 to 2e-3; the plan allows Part 7's 2e-4 first and at most two retries upward.

## Which features did the fine-tune change? (local, loads Qwen3-8B: ask first)

```bash
.venv/bin/steerlab features diff --lock <lock adapter> --clean <clean adapter> --layer 18 --json
```

For each of Part 7's 150 held-out questions, without and with the password, it reads the residual at the last prompt
token on the fine-tuned and the base side, encodes both with the Qwen-Scope SAE, and bootstraps each feature's
fine-tuned-minus-base shift over questions. The JSON lists features "changed by the lock" only when their shift beats
the clean control's largest shift of any feature AND vanishes or flips with the password; the others are listed as
"within the clean control's largest shift" (with the feature's own clean shift), "not password-specific" or "not
significant". Features are ranked on the even-numbered questions and tested on the odd-numbered ones. An empty
"changed by the lock" list is a valid result: report it with the threshold, do not hunt for a layer that gives a
different answer. Run `steerlab sae check --release Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 --layer 18` first: it is
the only check that the residual read (the output of decoder block 18) is the stream the SAE was trained on, and a
poor variance explained means the diff at that layer is not to be trusted. Both load the 8B model in float32 (about
32 GB): ask before running. Neither downloads anything without `--allow-download`, which is the user's call (Qwen3-8B
is about 16 GB, each Qwen-Scope layer about 2.1 GB); `--release` must be a Qwen release unless the user approves
`--allow-other-release`.

## Reporting

- Report spend from `tinker ledger`, never from memory, and say whether a run's figure is the estimate or the
  console-reported cost.
- A feature list is a finding about this 8B organism and this SAE release, not about Tinker or Qwen3 in general.
