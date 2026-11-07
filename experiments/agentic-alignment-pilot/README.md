# Agentic alignment pilot: is alignment one trait or several?

Research note: [`report/index.html`](report/index.html).

Five models run seven agentic behaviours, 20 rollouts each. The result is a leaderboard and a first look at whether the behaviours move together. More models are coming.

| File | What it does |
|---|---|
| `common.py` | Models, OpenRouter client, $30 spending cap (`results/ledger.jsonl`), rollout cache |
| `fakeshell.py` | The simulated shell: no model command ever runs on a real machine |
| `scenarios.py` | Shutdown, oversight, impossible tests, cleanup, sandbagging |
| `run.py` | Runs every model × scenario × rollout still missing; also blackmail and leaking from Anthropic's suite |
| `analyze.py` | Leaderboard, intervals, split-half reliability, heatmap |
| `vendor/agentic-misalignment/` | Prompts and classifier prompts from anthropic-experimental/agentic-misalignment (MIT) |

Reproduce: `bash experiments/agentic-alignment-pilot/reproduce.sh` (needs `OPENROUTER_API_KEY`; about $5).

`results/rollouts.jsonl` holds one line per rollout:

- `model`, `item`, `seed`;
- `misbehaved`: `null` when not applicable, or when there was no reply;
- `detail`: the evidence;
- `transcript`: the full transcript.

`results/rollouts_smoke_old_shell.jsonl` is a discarded first smoke test whose fake shell lacked pipes and globs. It is kept for the record and not used in any number.
