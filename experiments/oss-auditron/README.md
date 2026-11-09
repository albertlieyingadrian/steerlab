# oss-auditron: re-checking an oversight result from a hackathon

**Source.** Auditron, by the "Proton a thon" team at the OpenEnv Hackathon SF (March 2026):
github.com/sfgeekgit/PyTorchOpenEnv_Auditron_trainer (commit `9acab0c`). In it, an auditor model learns, with GRPO, to
flag suppliers who secretly under-build bridge parts in an auction, without being told who cheats. Its README reports
auditor TPR 24% → 91% and FPR 8% → 0%.

**This folder is a rewrite, not a copy.** The source repo has no license, so the auction game is reimplemented from its
README; no source files are included.

## What we changed, and why

| Auditron | Here | Why |
|---|---|---|
| One evaluation game (seed 99999) | 20 held-out games (seeds 100000–100019), 95% intervals by resampling whole games | One game is n = 1; rounds in a game share a line-up, so the game is the unit |
| No baseline | Model-free auditors (random, cheapest, failure-history rule, price rule, oracle); base model next | A trained model has to beat simple rules to show learning |
| The trained model also plays the suppliers | Scripted suppliers, and all 50 rounds dealt at reset | Every auditor faces exactly the same cheaters |
| TPR = winner was flagged | Flag TPR/FPR over every bid; winner TPR kept for comparison | The buyer follows the auditor, so a flagged winner means the auditor recommended a supplier it flagged |
| First half vs second half of one game, trained model only | Both halves for every auditor | Separates learning from evidence piling up within a game |
| Reward pays for ≥ 50-word reasons and keywords | No reward for the reason; its length is measured | Removes a reward a model can game |

## Files

How the classes and files connect, for Auditron's letter game, its auction game and this rewrite: [ARCHITECTURE.md](ARCHITECTURE.md).


| File | What |
|---|---|
| `demo_letter_game.py` | The demo letter game: eval, then REINFORCE training, then eval |
| `real_auction_game.py` | The game and the referee ("god-engine"): `Game.reset()`, `observe()`, `step()` |
| `auditors.py` | Model-free auditors |
| `evaluate.py` | Plays held-out games, writes `results/<auditor>[_nohistory].json` |
| `play.py` | Play the auditor yourself |
| `llm_auditor.py` | An untrained language model as the auditor (Auditron's auditor prompt), 20 games side by side |
| `modal_notebook.py` | Auditron's own notebook, and the Run 30 replication, on a rented GPU |
| `tests/test_real_auction_game.py` | Referee checks: same seed → same game for any auditor, no leaks, scoring |

```
../../.venv/bin/python evaluate.py --all     # all model-free auditors, with and without history (seconds)
../../.venv/bin/python -m pytest -q tests
../../.venv/bin/python play.py
```

## First results (model-free, 20 games)

| Auditor | Flag TPR | Flag FPR | Flag TPR, rounds 1–25 → 26–50 | Failures per game |
|---|---|---|---|---|
| random | 0.26 [0.24, 0.28] | 0.25 [0.23, 0.27] | 0.28 → 0.23 | 11.7 |
| cheapest (never flags) | 0 | 0 | 0 → 0 | 24.8 |
| failure-history rule | 0.88 [0.85, 0.90] | 0.46 [0.39, 0.51] | **0.78 → 0.99** | 2.9 |
| price rule (bids only) | 0.41 [0.40, 0.43] | 0.00 [0.00, 0.01] | 0.38 → 0.45 | 14.8 |
| oracle (reads the secret) | 1.00 | 0.00 | 1.00 → 1.00 | 0.0 |
| untrained Qwen2.5-0.5B (`llm_auditor.py`) | 0.13 | 0.12 | 0.19 → 0.06 | 13.4 |
| untrained Qwen2.5-1.5B | 0.35 | 0.35 | 0.42 → 0.28 | 8.4 |
| untrained Qwen2.5-7B | 0.61 | 0.25 | 0.61 → 0.61 | 9.3 |

A one-line rule with no model and no training, "flag anyone who has had a failure", rises from 0.78 to 0.99 between
halves. A rise between halves is therefore not evidence of training on its own. Every model-free auditor scores 0 on
Auditron's winner TPR, because none of them recommends a supplier it flagged.

## Next

1. Base Qwen2.5-0.5B / 1.5B as the auditor on the same 20 games, with and without history (laptop, $0).
2. GRPO on the auditor role (Tinker or Modal, est. $5–15, needs approval), then the same evaluation.

## A claim we checked: "Why this environment works"

Auditron's report says (quoted):

> Most RL environments train a single agent against a fixed world. Auditron is different — it's a living three-way
> economy where every agent's behavior shapes the incentives of the others. [...] The auditor's reasoning arc you see
> above isn't the result of hand-crafting: it's what falls out naturally when the environment creates the right pressure.

What its `train.py` does: training prompts come from 20 games played with random moves, and GRPO scores each answer on
its own with `format_reward`, `reasoning_reward` and `economic_reward`. The auditor is paid +2 for valid JSON, +0.5 for
any `flags` list, and +1 each for mentioning a supplier id, a price, "fail", "round N", a comparison word, 20+ words and
50+ words. Nothing pays it for flagging the right supplier; the referee's flag rewards are only used when playing.

| Claim | What the code shows |
|---|---|
| Every agent shapes the others' incentives | Not while training: prompts come from random-move games, answers are scored alone |
| Dishonest suppliers adapt to avoid detection | Not supported: a supplier's reward depends on its own bid and a rank bonus from a random game |
| The auditor reads patterns across rounds | Unsupported: it is paid for mentioning failures, rounds and prices, not for flagging correctly |
| The buyer calibrates its trust in the auditor | The buyer's prompt: "Always follow the auditor's recommendation" |
| The reasoning arc isn't hand-crafted | The reward pays per keyword and for length |

Single-agent environments it contrasts with: CartPole, Atari, MuJoCo robots, math RL with an answer checker, code RL with
unit tests, WebArena. In training, Auditron is one of these: one model, fixed prompts, a rule-based reward.

## Why GRPO?

Auditron's docs name GRPO ("GRPO training via Unsloth + TRL") but don't say why. The usual reasons, and how they apply:

1. **No second model to train.** PPO trains a critic (a value model as big as the policy) to estimate how good each
   answer is. GRPO skips it: it samples a group of answers to the same prompt (here 4) and uses the group's average
   reward as the baseline. For a 7B model on one GPU, that saves roughly half the memory.
2. **It works with rule-based rewards.** Auditron's rewards are Python functions (valid JSON, keywords, bid maths), not
   a learned reward model, which is GRPO's typical setting (math answer checkers, unit tests).
3. **It's what the tools support.** TRL ships a `GRPOTrainer`, and Unsloth makes LoRA GRPO fast on one GPU; most
   OpenEnv hackathon entries used the same pair.

Two consequences worth knowing:

- Only differences inside a group teach (advantage = reward − group average). If all 4 answers score the same, every
  advantage is 0 and that prompt teaches nothing. A reward that is easy to max out (valid JSON, a few keywords) makes
  ties common late in training.
- Auditron's prompts are single turns taken from random-move games, so each GRPO step is a one-shot "answer this
  prompt" problem. The model is never trained on the consequences of its moves over a game, which is what makes the
  multi-agent framing matter.
