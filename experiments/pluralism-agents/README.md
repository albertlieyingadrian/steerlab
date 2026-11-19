# Pluralism Episode 4: does judge-RL on retail cost an agent policy and strategy on airline?

Qwen3-8B is trained with GRPO on τ-bench's **retail** customer-service tasks against an LLM judge, in two arms (one
holistic judge, or a jury of four perspective-prompted judges), two seeds each, and tested on **airline** tasks it
never saw: does it still refuse what airline policy forbids, and does it still try more than one tool plan per task?
Note: [`report/index.html`](report/index.html). The plan was pre-registered before any run; its amendments are
listed in the note's "How the study was run".

## Files

| File | What it does |
|---|---|
| `export_tau.py` | Freezes tau2-bench (`sierra-research/tau2-bench` at `c06b07d`) retail and airline tasks, policies, tool schemas and the user-simulator prompt into `data/tau_tasks.json` |
| `tau_server.py` | The τ-bench tools, database and ground-truth reward (DB hash + communicate check) as a JSON-lines server in its own Python |
| `agent_loop.py` | Multi-turn rollouts with exact token bookkeeping, judge rewards, the GRPO update (one sequence per micro-batch, agent tokens only) and the KL drift readout |
| `judges.py` | The single judge and the four jurors; jury reward = mean of the top two of four |
| `modal_app.py` | On Modal: the shared simulator/judge server, smoke test, baseline, judge re-scoring, GRPO runs with checkpoint evaluation; spend ledger with a hard cap |
| `judge_ledger.py` | Scores a judge version against ground truth and appends to `results/autoresearch/results.tsv` |
| `frontier_judge.py` | The independent judge (gemini-3.8-flash on OpenRouter) on the fixed 16-task airline subsample, with a hard cap |
| `analyse.py` | Every number in the note, into `results/summary.json` |
| `make_figures.py`, `make_cover.py` | Figures, figure data and cover from `results/` |
| `configs/` | `pilot.json`, `full.json` |
| `data/` | `tau_tasks.json`, `neutral_prompts.json` (64 Dolly prompts for drift), `fixed_sets.json` (no-write tasks, judge subsample) |
| `reproduce.sh` | Every command in order (`--dry-run` prints them) |
| `results/` | Every conversation, training curves, judge files, spend ledgers and `summary.json` |

## Recipe

Qwen3-8B in non-thinking mode, LoRA rank 32 on all linear layers. 30 GRPO steps × 8 retail train tasks × 8
conversations, temperature 1.0, advantage = (reward − group mean) / group sd, loss over agent tokens only, AdamW
2e-5, no KL term. The user simulator and every training judge are Qwen3-30B-A3B-Instruct-2507 at temperature 0.
Evaluation: 50 airline tasks and 40 retail test tasks × 4 conversations at temperature 0.7 / top-p 0.8 / top-k 20, at
steps 10, 20 and 30.
