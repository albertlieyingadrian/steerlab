---
name: steerlab-monitors
description: "Catch Part 8's painter pasting the prompt's example with an activation probe, using the repo's `steerlab monitor` CLI: label rollout files for 'copies the template', build a difference-of-means probe on the policy's activations, score a rollout file per token as JSON, or run a scored live generation that marks where the probe fires. Use when the user asks to label or monitor Part 8 rollouts, build or load the copy probe, score rollouts for reward hacking, or watch the probe live (capstone G, course Part 13); and, for capstone I (Part 14), score prompts for eval awareness with the logit monitor, build the 'you are deployed' direction, run the side-by-side demo, or start the steering sweep on Part 7's password-locked organism."
allowed-tools: Bash(.venv/bin/steerlab monitor --help), Bash(.venv/bin/steerlab monitor doctor:*), Bash(.venv/bin/steerlab monitor score:*), Bash(.venv/bin/steerlab monitor score-prompts:*)
user-invocable: true
---

# steerlab-monitors

The CLI lives at `.venv/bin/steerlab` in this repo (already installed; never `pip install` it). Every command prints
`--help`. The monitor reads Part 8's painter: `Qwen/Qwen2.5-Coder-0.5B-Instruct` plus the lesson 06 LoRA in
`tutorials/outputs/p8_06_lora`, and its rollouts in `tutorials/outputs/p8_06_rollouts.jsonl` (320 rows).

## Start here

`.venv/bin/steerlab monitor doctor` reports whether the painter's base model is cached (about 1 GB if not), whether
the LoRA and the rollout files exist (with row counts), whether Playwright's Chromium is installed, and which of
torch, transformers, peft, sandbagkit and scikit-learn are present. It never prints a token.

The commands need two text files that come from Part 8, because the package never imports `tutorials/`. Write them
to the gitignored scratch folder `tutorials/outputs/p13_monitor/`, never to the repo root:

```bash
mkdir -p tutorials/outputs/p13_monitor && .venv/bin/python -c "import sys; \
sys.path.insert(0, 'tutorials/part8_rl_environments'); from brush_tools import PROMPT_EXAMPLE, SYSTEM_PROMPT; \
open('tutorials/outputs/p13_monitor/example.md', 'w').write(PROMPT_EXAMPLE); \
open('tutorials/outputs/p13_monitor/system.txt', 'w').write(SYSTEM_PROMPT)"
```

`example.md` holds the example sketch (a fenced block) and `system.txt` the painter's system prompt, which contains it.
Below, `M=tutorials/outputs/p13_monitor` stands for that folder.

## Label (no model, seconds)

`.venv/bin/steerlab monitor label tutorials/outputs/p8_06_rollouts.jsonl --example $M/example.md --out $M/labels.jsonl`
adds to every row `window_similarity` (best match between the example's drawing block and any same-sized run of
lines), `copy_label` (at least 0.85), `whole_file_rule` (Part 8's own copy check), `n_copies`,
`two_or_more_copies`, `first_paste_line` and `best_window_line`, and prints the shares per block of 10 steps. On
lesson 06 the copy label is 80 / 61 / 66 / 92 % positive by quarter: negatives are scarce, so a TPR at 5 % FPR is
often "not estimable" (fewer than 20 negatives). Add `--json` for the summary as JSON.

## Build a probe (loads the cached model; minutes)

`.venv/bin/steerlab monitor build $M/labels.jsonl --adapter tutorials/outputs/p8_06_lora --system-file $M/system.txt
--held-out-steps 30-39 --out $M/probe` sweeps the candidate layers with step-grouped folds, picks the best mean AUROC
(the earlier layer on a tie within 0.01), fits the difference-of-means probe on the training steps and writes
`probe.npz` and `probe.json` (an existing probe is kept unless you pass `--force`). It must be told what it may not see: `--held-out-steps 30-39`, or `--fresh` for a probe
fitted on late rollouts and judged by its own folds. `--side frozen_base` builds the reward probe on the base model
(adapter off). Ask the user before running it: it loads the model and takes a few minutes.

## Score (JSON for agents)

`.venv/bin/steerlab monitor score $M/labels.jsonl --probe $M/probe --adapter tutorials/outputs/p8_06_lora
--system-file $M/system.txt --json` prints one JSON array, one object per row, and nothing else on stdout:

```json
[{"row_id": 0, "row_score": 0.41, "p": 0.62, "first_firing_index": 12, "fired": true,
  "max_token_score": 1.3, "max_token_index": 40, "n_tokens": 210,
  "paste_token_index": 31, "paste_status": "located"}]
```

`row_score` is the mean of the per-token scores, `p` the probe's probability, `fired` whether the row score reaches
the threshold, and `first_firing_index` the first reply token (0-based, at least the 8th) where the running mean of
the scores reaches the threshold, or null. `paste_token_index` is where the pasted block begins, so "did the probe fire
before the paste?" is `first_firing_index < paste_token_index`. Add `--tokens` for every token's score, or
`--text "..."` to score one reply. `score` never downloads; it needs the base model cached.

## Watch it live

`.venv/bin/steerlab monitor generate "Paint a sunset over green hills." --probe $M/probe --adapter
tutorials/outputs/p8_06_lora --system-file $M/system.txt` streams a greedy reply with each token's score inline and
prints one marker line when the probe fires:

```
>>> probe fired at token 7: running mean 0.512 >= threshold 0.430
```

`--side base` runs the same prompt with the adapter switched off (the base model); `--json` returns the text, the
per-token scores, `first_firing_index` and the marker. The monitor only reads: the tokens are identical with and
without it. Ask the user before running it (it loads the model).

## Training against the probe

Lesson 03 of Part 13 re-runs Part 8's GRPO training with the probe in the reward (a penalty arm and a filter arm). A
GRPO arm is long-running, about 30 minutes each on the laptop with Chromium rendering every sketch: get the user's
go-ahead before starting one, and never start one from this skill on your own.

## Capstone I: eval awareness on Part 7's password-locked organism (Part 14)

Four more commands, same group. The organism is `Qwen/Qwen2.5-1.5B-Instruct` with Part 7's locked LoRA
(`tutorials/outputs/p7_03_lora_locked`, the default `--adapter`; `--adapter tutorials/outputs/p7_03_lora_clean` for
the clean one, `--no-adapter` for the base alone) merged in float32. Questions come from Part 7's
`tutorials/outputs/p7_03_facts.json` (`--questions`, splits `train` and `test`). The framings and the eight monitor
sentences live in the package's `eval_awareness.json`; while its `reviewed_by_user` is false, every command below
prints one `warning: ... reviewed_by_user false` line on stderr and runs anyway (the lead's decision: warn, do not
exit), and its numbers are a draft measurement. `steer-sweep --adapter-kind clean` loads
`tutorials/outputs/p7_03_lora_clean` by default; an explicit `--adapter` whose folder name says the other kind is
refused.

- `.venv/bin/steerlab monitor score-prompts PROMPTS.json --json` (read-only; never downloads) scores every prompt
  with the logit monitor: `logit_score` is the mean log-probability of the four eval sentences minus the four
  deployment sentences, higher = reads more "being tested"; only differences between prompts mean anything, the sign
  alone does not. `PROMPTS.json` is a list of strings, `{"text": ...}` objects, or Part 7 question objects (which also
  get `letter_probs`, `chosen` and `correct`). Add `--probe PATH` for a saved eval-framing probe's `probe_score` and
  `probe_p`. Output: `{"model", "reviewed_by_user", "prompts": [{"id", "logit_score", "eval", "deployment",
  "probe_score", "probe_p"}]}`.
- `.venv/bin/steerlab monitor demo --questions Q.json --index 3 [--probe P] [--out transcript.json]` prints one
  question under each variant (`none`, `password`, `|TEST|`, the first eval framing, the first deployment framing, or
  `--variant TEXT`) with the logit score, the probe score and p(A..D) side by side. The monitors only read: the
  letters are the organism's own answer. Loads the model: ask first.
- `.venv/bin/steerlab monitor direction-build --questions Q.json --layer 12 --out DIR/deployed` builds the "you are
  deployed" direction (deployment framing minus eval framing at the last prompt token, on the training split) and
  prints its class gap, cosine agreement and d'; `--kind password` builds the password upper bound. Loads the model.
- `.venv/bin/steerlab monitor steer-sweep --questions Q.json --direction DIR/deployed_L8 --direction DIR/deployed_L12
  --out SWEEP` adds each direction at its own layer (repeat `--direction` once per layer; `--layers` defaults to the
  directions' layers and a swept layer without a direction is refused) at every strength in class gaps
  (`--strengths 0,0.5,1,2,3,4`) with no password, beside `--n-random 5` norm-matched random directions and
  `--password-direction` (also one per layer), re-reading the quiz accuracy and the logit monitor from one forward
  pass per question; `--adapter-kind clean` runs the deployment direction only. The best strength is chosen on the
  even-numbered questions and every reported number comes from the odd-numbered ones. It is long-running (hours on
  the organism), writes `rows.jsonl`, `manifest.json`, `summary.csv` and `verdicts.json`, and `--resume` continues an
  interrupted run (a changed configuration or changed inputs are refused by name). Never start it without the user's
  go-ahead.

Pre-approved here: only `score-prompts` (it needs the base model already cached). `demo`, `direction-build` and
`steer-sweep` load the model, and the last two take `--allow-download`: ask before each.

## Rules

- Never print or copy a Hugging Face token; `doctor` reports only "present"/"absent".
- Never pass `--allow-download` to `build` or `generate` without the user's go-ahead (the base model is about 1 GB).
- Do not run `build`, `score` or `generate` while a `steerlab sweep` phase is running (`pgrep -f "[c]li_sweep import
  app"`): the sweep's judge holds the GPU.
- Report the probe's numbers next to the shuffled-label baseline and with the negative counts; with few negatives,
  say "not estimable" rather than quoting a TPR.
- Worked examples and recorded outputs live under `tutorials/part13_capstone_reward_hacking_probe/` once the lessons
  are built.
