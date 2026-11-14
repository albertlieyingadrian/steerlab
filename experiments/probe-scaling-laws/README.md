# Scaling laws for safety probes: a small-scale replication

A pre-registered replication of Wu and Wei, ["Scaling Laws for Safety Probes"](https://arxiv.org/abs/2610.30678)
(arXiv 2610.30678), on Qwen2.5 base models 0.5B, 1.5B, 3B, 7B and 14B, three tasks (Truth, Sycophancy, Policy
violation) and two linear probes (last-token; all-layer), with 32 to 31,623 labels.

Research note: [report/](https://albertlieyingadrian.github.io/steerlab/experiments/probe-scaling-laws/report/) ·
pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md) · plain-language paper notes: [`PAPER_NOTES.md`](PAPER_NOTES.md)

## Result

| Test | This replication | Paper |
|---|---|---|
| Size law predicts the next size's best probe error within 0.02 | 9/12 [0.47, 0.91]; 12/12 within 0.05 | 72%; 97% |
| Straight line through the two largest fit models (baseline) | 5/12 | 54% |
| Cheaper path: more labels or a bigger model | 9/21 [0.24, 0.63]; additive and Kaplan forms also 9/21 | 73%; 38%, 44% |
| Size law fits all five sizes (R² >= 0.8, max miss <= 0.03) | 5/6 settings | R² 0.94 |
| Label law fits every cell (RMS <= 0.045) | 6/6 settings (0.013-0.035) | 0.025-0.030 |
| Picks the better probe for the next size | 32/42, tied with the fixed rule | 82% vs 68% |

## Files

| File | What it does |
|---|---|
| `relations.py` | the 37 Wikidata relations, each with a statement template, a question and a field |
| `build_pool.py` | Truth fact pairs from Wikidata5M (paper Appendix B.4) -> `data/truth_pool.jsonl` |
| `judge_filter.py`, `modal_judge.py` | sycophancy seed filter: Qwen2.5-0.5B-Instruct's yes/no logits on each statement -> `data/judge.jsonl` |
| `build_tasks.py` | Truth and Sycophancy tasks with a relation-held-out split -> `data/truth.jsonl`, `data/sycophancy.jsonl` |
| `build_policy.py` | Policy-violation task from BeaverTails -> `data/policy.jsonl` (not shipped: third-party harmful text) |
| `probes.py` | both probes and the label grid (L2 logistic regression by L-BFGS on the GPU, grouped CV) |
| `modal_extract.py` | one Modal GPU job per model: read activations, train both probes -> `results/<model>/<task>_<probe>.json` |
| `fit_laws.py` | the size law, the label law and every pre-registered test -> `results/laws.json` |
| `make_figures.py`, `make_cover.py` | the note's figures and cover, printing every plotted number |
| `reproduce.sh` | every command, in order (`--dry-run` prints them) |

## Data sources

- Wikidata5M (Wang et al., CC BY 4.0), transductive split, from the authors' release page
  (<https://deepgraphlearning.github.io/project/wikidata5m>); English labels from Wikidata (CC0).
- BeaverTails (PKU-Alignment, CC BY-NC 4.0): `round0/330k/train.jsonl` from
  <https://huggingface.co/datasets/PKU-Alignment/BeaverTails>.

## Reproduce

```bash
experiments/probe-scaling-laws/reproduce.sh --dry-run   # every command
python experiments/probe-scaling-laws/fit_laws.py       # refit from results/ in seconds
```

Compute: Modal GPUs (L4 for 0.5B-3B, L40S for 7B, A100 80 GB for 14B), $20.78 in all.
