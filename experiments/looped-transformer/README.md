# looped-transformer: a small Dory-style looped transformer on chain following

Research note: [`report/index.html`](report/index.html). Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md).

NVIDIA's Dory runs one block of layers several times per token, with the input re-added each loop, and lets the user
pick the loop count at inference time. This folder rebuilds that idea at 0.8M parameters on a task where depth
plainly matters: follow a chain of assignments (`c=b`, `b=a`, `a=7`, shuffled among distractors) to the digit at its
end. It compares a looped model with a standard model of the same parameter count and one of the same compute,
scores every loop count from 1 to 16 on chains up to three times longer than any seen in training, and probes the
hidden state after every layer and loop.

| File | What it does |
|---|---|
| `common.py` | The task generator (48 shuffled assignments, 96 names), the standard and looped models, the shortcut baseline |
| `train.py` | Trains one model (curriculum to 8 hops, or `--max-hops 16`) and scores it at every hop count and loop count |
| `modal_train.py` | Runs `train.py` and `inside.py` on Modal (L4, at most 3 containers, no keep-warm) |
| `inside.py` | Linear probes after every layer and loop: pointer reach, digit frontier, answer by hops, convergence; demo-chain data |
| `analyse.py` | Every number in the note and the pre-registered verdicts → `results/summary.json` |
| `make_figures.py`, `make_demo.py`, `make_cover.py` | Figures, the interactive figure's data and the cover, from `results/` |
| `reproduce.sh` | Every command behind the note (`--dry-run` prints them) |

Results: `results/runs/<model>_s<seed>.json` (accuracy per loop count and hop count, training curve, curriculum),
`results/inside_<run>.json` (probe results), `results/summary.json`, `results/runs/modal_ledger.jsonl` (GPU seconds
per job). Checkpoints (`*.pt`) are not committed; `reproduce.sh` retrains them.
