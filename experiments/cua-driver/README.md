# cua-driver: an open macOS computer-use driver on Cua Driver Bench

An open driver built from the ideas Aside describes (accessibility tree, background-first input, one JavaScript
`repl` tool, a diff after every action), run on Scale's exact AgentEnv setup for Aside's 88/90 with Claude Opus 5.5,
beside a same-day cua-driver control. Result: 49/90 against cua-driver's 67/90 (McNemar 1 vs 19, p = 4e-5).
Note: [report/index.html](report/index.html).

| Path | What it is |
|---|---|
| `driver/` | The driver, Swift (`swift build -c release`): `AX.swift` reads the tree, `Background.swift` posts input to an app's process, `State.swift` keeps ids and diffs, `REPL.swift` is the JavaScriptCore sandbox, `main.swift` the MCP tools |
| `agentenv/` | The driver as an AgentEnv arm: `envs/ourdriver/` (driver.py, env.toml, pinned mcp.json and sha256) and `sync.py`, which builds the binary and installs the arm, its computers, tasks and schedules into a clone of Scale's port |
| `tools/run_par.sh`, `tools/run_seq.sh` | Run a schedule of AgentEnv tasks, one VM per runner; harness failures retried with the same trial id and logged |
| `tools/collect_trials.py` | Archives each trial (grader result, transcript, snapshots) through AgentEnv's records and writes `trials.csv` |
| `analyse.py` | Every number and figure in the note, from `results/trials.csv` |
| `make_cover.py` | The cover, from `results/trials.csv` |
| `reproduce.sh` | Every command, in order (`--dry-run` prints them) |

Results (`results/`): `trials.csv` (180 final trials, columns in the note's appendix), `summary.json`,
`per_variant.csv`, `calibration.tsv` (18 cua-driver 0.47.10 trials), `dev_s5-haiku-dev.csv` and
`dev_s5b-haiku-dev.csv` (development trials, Haiku 5.5).

Frozen driver: 0.5.1-step5, binary sha256 `6278a5b32091093c9d3deb1d25541fa9e0d59d2a2ca7e781ffe58642b6162fdc`.
The binary itself is not committed; `agentenv/sync.py` builds it from `driver/`.
