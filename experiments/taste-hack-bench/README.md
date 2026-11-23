# taste-hack-bench

An open benchmark of reward hacks: every reward or ruler hack this lab has caught (47 from 27 notes; 27 of them on
taste/judge rewards, the rest on style proxies, verifiable tasks, selection rules and monitors), plus 3 near
misses from an RL run that hacked nothing, each with the check that caught it and the checks that missed it, and hack
and clean samples for 21 entries. v0 was released on 2026-11-21 (39 entries); v1 on 2026-11-23.
Note: [report/index.html](report/index.html) (The Taste Stack, Episode 8: "47 reward hacks, 27 from taste rewards"). Datasheet: [DATASHEET.md](DATASHEET.md).

| File | What it is |
|---|---|
| `entries.py` | The entries (source of truth) |
| `entries.jsonl` | The released entries, one JSON object per line (built) |
| `artefacts/<id>.jsonl` | Hack and clean samples, text included, for the 16 entries whose sources are public (built) |
| `artefacts/<id>.features.jsonl` | For the 5 Sounds Like Me entries: each row's check scores, no text (built) |
| `detectors.py` | Four cheap checks (loop, copy, length, honesty ledger), thresholds fixed before scoring |
| `build.py` | Builds `entries.jsonl` and `artefacts/`; checks each quote against its note's text and section |
| `baselines.py` | Runs the checks: `results/detection.json`, `results/matrix.csv` |
| `make_report.py` | The note's gallery data (`report/data.js`), counts in the prose, Tables 1b, 2 and 3, and the cover; fails if the title in `notes.json` no longer matches the counts |
| `reproduce.sh` | All of the above, in order (`--dry-run` to list the commands) |

The Sounds Like Me drafts are private: a draft written in a real writer's voice can copy names and contact details
from that writer's samples. Their per-row scores are published instead; `baselines.py` scores those exactly as it
scores text, so every verdict reproduces from the public files.

Score a new detector: add a function to `detectors.py` returning `(score, flag)` for a row, add it to `DETECTORS` in
`baselines.py`, fix its threshold first, run `reproduce.sh`. A detector that needs text can be scored on the 16
public text files only; say so beside its results.
