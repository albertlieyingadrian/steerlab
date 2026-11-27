# repl-gram-modular-pretraining

A toy-scale replication of "Modular Pretraining Enables Access Control" (GRAM; arXiv 2607.08077, code
github.com/agencyenterprise/modular-pretraining): the paper's own 26M-parameter Simple Stories experiment, rebuilt
from its released code, with GRAM, data filtering, MaxEnt unlearning and the relearning attack, on three seeds.
Research note: [`report/index.html`](report/index.html). Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md).

| File | What it is |
|---|---|
| `gram.py` | Model (the paper's dense Transformer plus auxiliary MLP modules), data streams, the pretraining loop with one AdamW per parameter partition, the GRAM batch schedule (`do_routed_unordered`, equal compute), filtering, MaxEnt, and the relearning attack |
| `modal_app.py` | Modal jobs: `prep` (tokenise Simple Stories), `smoke`, `launch` (one H100 per seed), `pull` (copy result JSON) |
| `analyze.py` | Compute ratios, seed means with 90% and 95% CIs, the pre-registered tests; writes `results/summary.json`, `results/table.tsv` |
| `make_figures.py` | The note's figures from `results/`; prints every number it plots |
| `reproduce.sh` | Every command behind the note (`--dry-run` prints them) |
| `results/runs/seed_N/` | Per seed: `baseline.json` (steps, loss curve, final val/test loss per topic), `gram_eval.json` (loss per profile), `gram_elicit.json`, `filter_<profile>.json`, `filter_<profile>_elicit.json`, `maxent_<profile>.json`, `log.txt` |
| `results/runs/maxent_choice.json` | The MaxEnt (lr, alpha) grid on seed 1 and the chosen setting |
| `results/summary.json` | Every number the note quotes |
| `results/table.tsv` | Long table: seed, method, profile, topic, class (retain / forget / elicited_forget / elicited_forget_8x), test loss, compute ratio, best elicitation round |
| `results/spend_ledger.jsonl`, `results/calls.jsonl` | Every Modal launch, with its estimate, and the spawned call ids |

Topic names: `core` is the 44 benign topics; the four restricted ("auxiliary") topics are the first four
alphabetically, as in the paper. A *profile* names the modules switched on: `core` (none) or `core+<topic>`.
