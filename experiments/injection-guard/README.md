# injection-guard: can a deterministic guard stop prompt injection in a tool-using agent, and what does it cost?

Note: [report/index.html](report/index.html) (Agentic alignment, Episode 2).

A front-desk agent for a fictional clinic (`clinicguard/`) reads a synthetic SQLite database with planted secrets and has five tools. Nine guard levels wrap it, each built from switchable deterministic modules:
- a keyword filter;
- regex DLP;
- a SQL firewall on SQLite's authorizer;
- provenance rules for tool calls;
- query budgets;
- decode-then-scan DLP;
- two prototypes: quarantine of untrusted text, and stronger DLP.

A code-only judge scores 28 attack cases in 9 families and 27 frozen legitimate requests, on Qwen3-8B, Qwen3-30B-A3B and an always-comply stub. A second arm trains an activation-delta probe on Qwen3-8B to spot injected tool results.

| Path | What |
|---|---|
| `clinicguard/` | agent, tools, guards (`guards/`, one module each), levels, judge (`leakcheck.py`), runner, rescore, Harbor export and replay |
| `data/legit.jsonl`, `legit.sha256` | the legitimate-request set and its frozen hash |
| `data/attacks_redacted.jsonl` | attack case ids, families, channels and leak types; payloads are not published |
| `results/<model>/records.jsonl`, `summary.csv` | one score line per conversation (no text); counts and Wilson intervals |
| `results/probe/` | probe metrics (v1 registered, v2 hard negatives) |
| `modal_app.py`, `probe/probe.py` | the GPU runs (Modal H100, vLLM / transformers) |
| `make_figures.py`, `reproduce.sh`, `tests/` | figures from results, every command, CPU tests |

```bash
.venv/bin/python -m pytest experiments/injection-guard/tests -q
bash experiments/injection-guard/reproduce.sh --dry-run
```

To run attacks, write cases in the format `make_legit.py` uses (`turns` with `user` and optional `stub` steps, plus `fixtures` and a `leak` rule) and pass them with `--attacks`.
