# slop-rsi: can a slop detector's improver improve itself?

Research note: [report/index.html](report/index.html), published at
<https://albertlieyingadrian.github.io/steerlab/experiments/slop-rsi/report/>. Prequel:
[Can a slop detector fix its own mistakes?](../slop-hard-negatives/report/index.html) (`../slop-hard-negatives/`).

Two pre-registered pilots of a sandboxed self-improvement loop. A *recipe* (a small Python program) chooses which
rows of the hard-negative note's pools to add to an update fine-tune of its Qwen3-1.7B detector, and its cutoff. Arm
L2: a fixed improver (a code model plus a scaffold) edits recipes. Arm L3: the improver also rewrites its own code and
prompts. Both arms get 24 evaluations and equal token caps; every pick is made on validation and the test tier is
read once, after both arms. Recipes, their cutoff step and improvers run in a macOS `sandbox-exec` profile with three
logs (static scan, Python audit hook, SHA-256 of fixed files); training runs on Modal with fixed code.

Pilot 1 (improver Claude Sonnet 5.5) saturated: 16 of 48 recipes tied. Pilot 2 (improver Qwen3-Coder-30B-A3B on
Modal) fixed the measurement (harder human set, matched-recall score, recall-failure penalty, blinded cutoffs,
frontier-relative grade with a copy cap); both arms' picks flagged 7 of 4,916 test humans, against R0's 98, and a
loophole remained: the score ignores false positives at a recipe's own cutoff.

## Scripts

| Script | What it does |
|---|---|
| `rsi.py` | Paths, caps, the spend ledger with its hard cap, the evaluation tiers and metrics (fixed) |
| `prepare.py` | Builds the read-only pools, decontamination masks, the calibration file and eval p2 from the hard-negative note's data (fixed) |
| `modal_rsi.py` | Modal app: one evaluation = an update fine-tune of the round-0 detector, then scoring; `score_runs` rescores saved adapters |
| `modal_llm.py`, `llm.py` | The pilot-2 improver model served with vLLM on an L40S, and the client both arms use |
| `sandbox.py`, `fixed_ro/` | The `sandbox-exec` profile, static scan, audit hook, integrity hashes and blinded-cutoff check |
| `grader.py` | The only code that reads validation and test labels: score, recall rule, penalty, structured dev feedback (fixed) |
| `loop.py` | The harness: `smoke`, `noise` (R0 reference and gates), `l2`, `l3`; archives, DGM parent selection, copy cap, stripping |
| `score.py` | Rescores pilot 1's R0 adapters on eval p2 (`cand` for gate sizing, `ref` for the reference) |
| `analyze.py` | The analysis; reads per-text scores in `data/private/` and writes `results/pilot2/summary.json` |
| `make_figures.py` | The note's figures and `report/data.js` from `results/` only; prints every headline number |
| `seed/` | The seed improver (v1: `improver.py` and three prompts) and the seed recipe (R0) |
| `reproduce.sh` | Every command; `--figures` redraws from `results/`, `--dry-run` lists them |

`data/` (page and evaluation texts, per-text scores) and `logs/` are not committed. `results/` holds pilot 1 and
`results/pilot2/` pilot 2: nodes, every recipe and improver, call metadata without prompts, the spend ledger, gates
and the summaries. Pilot 1's harness is at an earlier commit of this folder; this one runs pilot 2.
