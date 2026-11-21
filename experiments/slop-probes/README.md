# slop-probes: tiny probes vs open VLM judges at telling slop from purpose-made web design

The Taste Stack, Episode 3. On Ep 2's design index (`../design-index/`), probes on frozen features (screenshot
embeddings, the page's block tree, its layout geometry) are compared with two open vision-language judges at telling
pages on AI site builders from human sites. Every arm is checked against the builder's badge, page statistics and a
left-out builder. A probe trained only on pages generated from a written taste constitution
(`constitution.md`) tests the Constitutional Classifiers recipe for design.

Plan, fixed before any held-out score: the plan. Note:
`report/index.html`.

## Pipeline

| Step | File | Where | Output |
|---|---|---|---|
| Masked screenshots and badge-free page features, from Ep 2's renders (read-only) | `modal_masked.py` | Modal A10G | `~/.cache/steerlab/slop-probes/masked.pkl` |
| Constitution pages: generate, render, featurise | `modal_constitution.py` (`::generate`, `::render`, `::features`) | Modal H100, CPU, A10G | `constitution.pkl`, `cst_meta.jsonl` |
| VLM judges (dev: both prompts; held: the prompt chosen on dev) | `modal_judge.py` | Modal H100 | `judge/<model>.jsonl` |
| Probes, baselines, shortcut checks, rulers, combinations | `probes.py --stage dev` then `--stage held` | laptop | `results/dev/*.json`, `results/*.json` |
| Figures, cover and the interactive's data | `make_figures.py` | laptop | `report/figures/`, `report/data.js` |

Shared pieces: `ep3.py` (labels, splits, the badge box, a loader for Ep 2's index), `budget.py` (Modal $40 soft,
$100 hard, from Modal's billing API). Every try on dev is in `results.tsv`.

## Results files

| File | What it holds |
|---|---|
| `results/probes.json` | Each probe and baseline: chosen C, dev AUROC, held-out AUROC [95% CI], within-industry AUROC, known-answer AUROC |
| `results/judges.json` | Each judge: dev AUROC per prompt, held-out AUROC masked and unmasked, score histogram, parse share |
| `results/judge_rulers.json` | the judge known-answer check (all splits) and the judge-agreement check (the two judges on real held-out pages) |
| `results/comparison.json` | Paired probe − judge differences, and P1 |
| `results/combos.json` | Probe + judge and constitution + judge combinations |
| `results/shuffle.json` | Probes trained on dev labels shuffled within industry (10 shuffles) |
| `results/lobo.json` | Leave one builder out |
| `results/constitution.json` | The constitution probe: render pass rate, the brief-3 separability check, held-out and known-answer AUROC, repetition of each synthetic class |
| `report/data.js` | Held-out scores per arm and the label, shuffled order, no page ids or domains |

## What is not in the repo

Screenshots, page HTML, domain lists, the generated pages and per-page scores keyed by page id stay in
`~/.cache/steerlab/` and on Modal Volumes. Builders are numbered, not named. The curated-library class ships derived
numbers only.
