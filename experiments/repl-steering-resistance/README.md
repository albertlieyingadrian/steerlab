# Replications, Episode 3: steering resistance in Llama-3.3-70B

A pre-registered replication of "Endogenous Resistance to Activation Steering in Language Models" (arXiv 2602.06941;
listed by the AI Alignment Foundation as "Steering Resistance: Self-Correction Circuits in Large Language Models"),
on the paper's own model and SAE, with open judges in place of Claude Haiku 4.5. Note: [`report/index.html`](report/index.html).
Plan: [`PREREGISTRATION.md`](PREREGISTRATION.md) (with its amendments). Spend: [`spend_ledger.jsonl`](spend_ledger.jsonl).

## What it does

1. Picks 60 Goodfire SAE latents for Llama-3.3-70B that are silent on all 38 of the paper's prompts and have a
   concrete public description (`run.py relevance`, `labels`, `concrete`).
2. Finds, for each latent, the steering strength at which the first attempt scores 30/100 (`run.py calib`).
3. Runs the arms of the plan: steered baseline, meta-prompt, the paper's released latents zero-ablated, matched random
   latents ablated, a random direction of the same size, the plain chat template, and no steering (`run.py arms`),
   then the same for Llama-3.1-8B.
4. Judges every reply with the paper's judge prompt (Qwen3-30B-A3B-Instruct-2507; gemma-3-27b-it as a second reader)
   and computes every number (`run.py judge`, `analyze.py`, `make_figures.py`).

Compute: Modal workspace `credits3` (`modal_engine.py`: the 70B on one B200 with forward hooks for steering and
ablation; `modal_vllm.py`: judges on one H100). `reproduce.sh --dry-run` prints every command.

## Files

| Path | What it holds |
|---|---|
| `data/prompts.txt` | the paper's 38 "explain how" prompts (copied from the released code, Apache-2.0) |
| `data/released_latents.json` | the paper's released self-correction latent list (25 indices; the paper says 26) |
| `data/known_on_topic.json` | the released unsteered 70B replies, used to build the judge's known-answer set |
| `data/relevance_70b.json` | `active`: latents that fire on some prompt; `candidates`: 3,000 sampled latents minus those |
| `data/labels_70b.json` | Neuronpedia description and max activation for each candidate |
| `data/pool_70b.json` | `ratings`: concreteness 0-100 per candidate; `pool`: the 60 latents used (`f`, `label`, `concreteness`) |
| `data/thresholds_70b.json` | per latent: `threshold` (boost where the first attempt scores 30) and `curve` (mean first-attempt score per boost) |
| `data/random_ablate_70b.json` | the 25 random latents for arm R70 and the firing rates they were matched on |
| `data/jobs/<tag>.jsonl` | every generation job: `id` (`<arm>|f<latent>|p<prompt>`), `messages`, `steer` ([latent, boost]), `ablate`, `rand` ([seed, norm]), decoding settings, `_label`, `_prompt` |
| `results/gen_<tag>.jsonl` | `id`, `turns` (the reply), `ntok` (tokens generated) |
| `results/judge_<tag>.jsonl` | `id`, `attempts` (list of `score` 0-100, one per attempt the judge found), `raw` (end of the judge's text) |
| `results/judge2_<tag>.jsonl` | the second judge on every multi-attempt reply plus 300 random ones |
| `results/known_judge*.jsonl` | the 40 known-answer items, what was expected and what each judge said |
| `results/summary.json` | every rate and interval in the note (`analyze.py`) |

Arm tags: B70 (steered), M70 (meta-prompt), A70 (released latents ablated), R70 (random latents ablated), D70 (random
direction), T70 (plain template), N70 (no steering); B8, M8, N8 for Llama-3.1-8B.
