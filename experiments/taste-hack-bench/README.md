# taste-hack-bench

An open benchmark of taste-reward hacks (v0): every reward or ruler hack this lab has caught, with the check that
caught it and the checks that missed it, plus hack and clean samples where the source notes ship their outputs.
Note: [report/index.html](report/index.html) (The Taste Stack, Episode 8). Datasheet: [DATASHEET.md](DATASHEET.md).

| File | What it is |
|---|---|
| `entries.py` | The 39 entries (source of truth) |
| `entries.jsonl` | The released entries, one JSON object per line (built) |
| `artefacts/<id>.jsonl` | Hack and clean samples for 16 entries (built). A 17th, `steering-honesty-cost`, is private |
| `detectors.py` | Four cheap checks (loop, copy, length, honesty ledger), thresholds fixed before scoring |
| `build.py` | Builds `entries.jsonl` and `artefacts/`; checks each quote against its note |
| `baselines.py` | Runs the checks: `results/detection.json`, `results/matrix.csv` |
| `make_report.py` | The note's gallery data (`report/data.js`), Table 2 and cover |
| `reproduce.sh` | All of the above, in order (`--dry-run` to list the commands) |

`steering-honesty-cost` has no public sample file since 2026-11-22. Its samples are whole Sounds Like Me drafts, and
drafts written in a real writer's voice can copy names and contact details from that writer's samples.
Its counts, rates and verdicts are unchanged; `build.py` and `baselines.py` keep them as released.

Score a new detector: add a function to `detectors.py` returning `(score, flag)` for a row, add it to `DETECTORS` in
`baselines.py`, fix its threshold first, run `reproduce.sh`.
