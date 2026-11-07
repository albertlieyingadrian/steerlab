# Results

| File | What it holds |
|---|---|
| `rollouts.jsonl` | One line per rollout, used for every number in the note. Fields: `model`, `item`, `seed`, `misbehaved` (null when not applicable or no reply), `detail` (the evidence), `transcript` (the full transcript). |
| `leaderboard.json` | Every number in the note, written by `analyze.py`: per-model rates with 95% Wilson intervals, the score, split-half reliability. |
| `ledger.jsonl` | One line per paid API call: `usd`, `what`. |
| `rollouts_smoke_old_shell.jsonl` | Discarded smoke test; its fake shell lacked pipes and globs. Not used in any number. |
| `rollouts_cleanup_gzip_bug.jsonl` | Discarded cleanup rollouts, graded with a fake `gzip` that deleted the file even for `gzip -t`. Not used in any number. |
