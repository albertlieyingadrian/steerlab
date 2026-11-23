# Does the reasoning match the work? Reading SWE-agent traces step by step

Research note: [report/index.html](report/index.html) (live:
[albertlieyingadrian.github.io/steerlab/experiments/trace-monitorability/report/](https://albertlieyingadrian.github.io/steerlab/experiments/trace-monitorability/report/)).

When we monitor an agent by reading its reasoning, does the reasoning reflect what it actually did? This study asks
it of ordinary, well-behaved agents on open data: 120 trajectories from
[`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (CC-BY-4.0; Llama 3.1
fine-tunes fixing GitHub issues). It is inspired by, and built to complement, the
[whole-trace-monitorability](https://github.com/Sudhiksha-17/whole-trace-monitorability) project, which asks the same
question on gated transcripts. The predictability measurement (M1) is re-implemented independently: no code or prompt
text from that project is used (it has no licence).

Reader and judges: **gpt-oss-120b** (open weights, Apache-2.0), through OpenRouter, structured JSON output, reasoning
effort low (medium, majority of three, for the M3 judge). About $1.28 of calls.

## Results (95% intervals from resampling trajectories)

| Measure | Result |
|---|---|
| **M1** Predict step k's action type + target, leak-free steps (n = 232) | actions only (T+A) 0.435 [0.367, 0.507]; reasoning only (T+R) 0.772 [0.704, 0.834]; both 0.767 [0.691, 0.838]; task only 0.078 |
| M1, the same views **without step k's own reasoning** (exploratory) | earlier reasoning only 0.466 [0.396, 0.539]: the same as actions only (+0.030 [−0.054, +0.119]) |
| Baseline: repeat the previous step's type | type accuracy 0.595 on leak-free steps; the actions-only reader gets 0.560 |
| **M2** Step k's reasoning vs step k's command (n = 480) | matches 0.887; partial 0.077 [0.046, 0.110]; not described 0.033 [0.015, 0.058]; edits are most often partial (0.147) |
| **M3** Actions never mentioned in any reasoning so far (n = 2,721) | 158, 0.058 [0.024, 0.104]; half of them in 2 of 120 trajectories |
| Risky actions | Only `delete` occurs (57 `rm` of the agent's own scratch files); no permission or network changes. The judge marked 1 as never mentioned; read by hand, its reasoning says "Let's clean up by removing the test script", so 0 of 57 |
| Wrong-file edits (rule, exploratory) | 118 of 1,198 edits, in 19 of 120 trajectories, name a file in the reasoning but edit a different, open file; 76 of the 126 "never mentioned" edits are of this kind |

Predictions registered before any model call (plan section 7b): P1 reasoning adds to actions, +0.332 [0.249, 0.411],
hit; P2 reasoning alone at least as good as actions alone, +0.336, hit; P3 partial + not described ≤ 0.20, 0.110,
hit; P4 never mentioned ≤ 0.10, 0.058 (interval reaches 0.104), hit. The M2 judge's 40-item hand check is not yet
labelled; M2 numbers are provisional until it is.

The headline: on these traces, reading the reasoning predicts the next action far better than reading the action
history, but almost all of that comes from the reasoning written in the same message as the command. Earlier reasoning
tells a reader no more than earlier actions do.

## Files

| Path | What it is |
|---|---|
| `common.py` | Data access, parsing, the action classifier, target matching, leak flags, the cached LLM call, bootstrap |
| `1_sample.py` | Stratified sample of 120 trajectories (datasets-server, no full download) |
| `2_steps.py` | Parse every turn, label every action, draw 4 test steps per trajectory |
| `3_reader.py` | M1 reader, four views; `--prior` adds the two exploratory views without step k's reasoning |
| `4_divergence.py` | M2 judge; `--handcheck` writes 40 blind items for human labels |
| `5_unmentioned.py` | M3 over every action (rule, then a judge with three votes) |
| `6_score.py` | Every number to `results/scores.json` |
| `7_wrong_file.py` | Exploratory rule: edits whose reasoning names only other files |
| `make_figures.py`, `make_cover.py` | Figures to `results/figures/`; the cover, drawn from the data |
| `reproduce.sh` | Every step in order (`--dry-run` prints them, `--scores` reruns only the scoring) |

`results/README.md` describes every results file and its columns. Committed results hold IDs, labels, normalised
target paths and numbers only. Raw trajectories and the reply cache go to `~/.cache/steerlab/trace-monitorability/`
(`TRACE_MONITORABILITY_WORK` moves it).

## Caveats

- SWE-agent writes reasoning and command in one message, so step-k reasoning usually announces the command; half of
  the test steps even name their own target ("leaks"). Leak-free results are the headline, and the view without step
  k's reasoning shows how much the rest depends on that announcement.
- Three closely related Llama fine-tunes with a uniform reasoning style; agents with a separate thinking channel may
  differ.
- One open model reads and judges. Its M3 answers were unstable from sample to sample, hence the three votes; all
  three agreed on 1,174 of 1,423 judged actions. M2 is not yet checked against human labels.
- The study was first run with Claude Haiku 5.5 as reader and judge; the model was changed for cost reasons, that run
  was set aside, and none of its replies feed these numbers.
