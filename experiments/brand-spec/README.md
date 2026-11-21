# Can a brand book become a checkable spec?

Code and data for the steerlab note `report/index.html` (The Taste Stack, Episode 5). Two public design-system guides
(the GOV.UK Design System and the U.S. Web Design System) are turned into machine-checkable specs, every rule traced
to its source line, with exceptions and the readings the text leaves open marked. A checker reads rendered pages and
writes a proof artifact per page; a copy check tells near-copies of the brand's own pages from new on-brand work; and
two open LLMs are tested on the same pages, rule identification apart from rule application.

Run everything with `reproduce.sh` (`--dry-run` prints the commands). The generation step needs a Modal account; the
rest runs on a laptop with Playwright's Chromium.

## Files

| File | What it does |
|---|---|
| `vendor/` | The two brand books as plain-text lines with their upstream file and line (`source.tsv`), the builder and the licences; see `vendor/README.md` |
| `formalise.py` | Stage 1: builds `spec/<brand>_spec.json` (tokens, components, rules, residue, a kind for every line) and lints it |
| `pages.py` | The page generator: brand components to HTML with inline token values; planted violations and exception traps |
| `extract.js`, `render.py` | Stage 2: render a page in headless Chromium at 1280 px and read its element facts from computed styles |
| `checker.py` | Stages 3-4: evaluate a spec on a page's facts and return the proof artifact |
| `report/checker.js` | The same checker in JavaScript, for the note's interactive |
| `dataset.py` | Builds, renders, extracts and checks every page set; freezes them (`data/manifest.json`) |
| `questions.py` | The model test sets: page audits (T1), rule identification (T2), rule application (T3, and T3c with the ratio given) |
| `ruler.py` | Every pass bar: checker vs planted answers, extractor vs intended facts, JS vs Python, contrast pairs, the magic-number claim, the parser; `--real` checks the two brands' home pages |
| `copycheck.py` | The copy check: structure, layout, computed-style and text similarity to the brand's existing pages, token overlap as the cheap baseline |
| `prompts.py`, `modal_gen.py`, `parse.py` | The prompts; generation with Qwen3 under vLLM on a Modal H100 through a spend ledger with a cap; the reply parser |
| `score.py` | Scores every reply, adds the baselines, writes `results/summary.json` |
| `make_figures.py`, `make_demo.py`, `make_cover.py` | Figures, the interactive's data (`report/demo.js`) and the cover, all from `spec/` and `results/` |
| `results.tsv` | Every dev tuning try (autoresearch ledger) |

## Data files

| File | What it holds |
|---|---|
| `data/pages_<brand>_<split>.jsonl` | Compliance pages: `id`, `kind` (on, trap, off), `domain`, `page_type`, `recipe` (sections, text seeds, style knobs), `labels` (the planted `[element, rule]` pairs; `[]` on compliant pages), `planted` (name, rule, whether the text settles it, element), `traps`, `html` |
| `data/copy_<brand>_<split>.jsonl` | Copy set: references (`ref`), near-copies (`near`, with `ref` and `strength` 1-3), new designs (`new`), remixes (`remix`, held-out only) |
| `data/demo_govuk.jsonl` | The interactive's page and its single-change versions |
| `data/facts_<file>.jsonl` | The extractor's facts for every page, same order: per element `id, role, tag, text, region, form, form_x, x, y, w, h, color, bg, eff_bg, parent_bg, border, font, size, weight, line_height, margin, padding, underline, disabled` |
| `data/intended.jsonl.gz` | What the generator meant to render, element by element (the extractor's reference labels) |
| `data/t1_*`, `t2_*`, `t3_*`, `t3c_*` | The model test sets (prompts are built from these by `prompts.py`) |

## Results files

| File | What it holds |
|---|---|
| `results/raw/<task>_<brand>_<split>_<model>_<arm>[_s1].jsonl` | One reply per item: `text` (thinking included), `tokens`, `prompt_tokens`, `finish` (`length` = hit the token cap) |
| `results/scored/<same>.jsonl` | Per item: for T1 `claims`, `exact`, `verdict`, `el_exact`, `strict_hits`, `missed`; for T2 `cited`, `strict`, `loose`, `random_citation`; for T3 `pred`, `correct` |
| `results/summary.json` | Every rate as k/n with a 95% bootstrap interval, the baselines, and the predictions' comparisons (`summary_dev.json` for dev) |
| `results/proofs_<file>.jsonl` | The checker's verdict, failures and counts per page (full artifacts for the demo set) |
| `results/copy.json` | The copy check: per-page component scores and nearest reference, AUROCs with intervals, the gate chosen on dev and its held-out accuracy |
| `results/ruler.json` | The pass-bar checks, the magic-number claim and the real-page check |
| `results/spend.jsonl` | The Modal spend ledger |
