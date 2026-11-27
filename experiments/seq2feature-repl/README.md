# seq2feature replication and extension

Replicates and extends [seq2feature](https://github.com/patrickod32/seq2feature) (commit `f0616ac`): a 5.3 MB
text-only probe distilled from a Gemma-2-9B GemmaScope SAE (layer 20, 16k latents, 2,048 kept). The note is in
`report/index.html`; the plan, fixed before any scored run, is `PREREGISTRATION.md`.

| File | What it does |
|---|---|
| `common.py` | Loads the released bundle and probe (fp32, shipped int8, int4 as notebook 01 makes it), scoring, bootstraps |
| `make_sample.py` | Draws E2's 1,000 held-out spans (200 per domain, seed 0) into `data/e2_sample.jsonl` |
| `modal_harvest.py` | Gemma-2-9B + SAE on one Modal A100-80GB: E2's six input conditions and notebook 03's OOD spans |
| `score_cpu.py` | Probe, bag of words and bge-small scores for every test span (and the OOD spans) on CPU |
| `analyze.py` | Every number in the note, written to `results/*.json` |
| `make_figures.py` | The figures in `report/figures/`, printing the numbers it plots |
| `reproduce.sh` | All of the above in order (`--dry-run` prints the commands, `--gpu` re-harvests) |
| `spend_ledger.jsonl` | Every Modal call with its worst case and its wall-clock cost |

`work/` (not committed) holds the scores and readouts: about 400 MB. `results/` holds the summaries the note quotes.
