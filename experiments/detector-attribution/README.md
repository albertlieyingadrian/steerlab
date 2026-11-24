# detector-attribution: which words and inside features make an AI-text detector say "AI", and do they move Pangram?

The note: [report/index.html](report/index.html) (Model voice, Episode 7). Pangram is a closed API, so nothing
white-box can run on it. This experiment reads four open detectors from the inside (word occlusion, integrated
gradients, Qwen-Scope SAE features), reads the model that wrote the AI text with its own SAE, and then tests whether
what the open detectors rely on moves Pangram, by editing the text: deleting the words and sentences they lean on
(against random deletions of the same size), adding stock "AI tells" to human passages (against neutral edits at the
same places), and splicing human and AI halves to compare where each detector locates the AI part.

Texts: deslop-h2h's Qwen3-8B replies (six fluent arms, 531 cut texts) and deslop's 316 human passages (Project
Gutenberg, Wikipedia), each cut to 120-195 words at a sentence end. Split by prompt (and by passage) into discovery
and confirmation halves: lists and features are chosen on discovery texts, every test runs on confirmation texts.

## Scripts, in order

| Script | What it does |
|---|---|
| `common.py` | The corpus (cut texts, splits), word units, sentence spans, deletion, bootstrap |
| `budget.py` | Hard caps in code: Pangram $60, Modal $80; every paid call is logged to `results/spend.jsonl` |
| `modal_attr.py` | Modal app `detector-attribution`: `Det(name)` scores, occludes, runs IG and (9B) SAE attribution and ablation; `Gen` reads Qwen3-8B with its layer-18 SAE |
| `attribute.py` | `score`, `words` (occlusion, the primary attribution), `occlude` (9B sentences), `deletion` (top vs 20 random), `ig` (160-text subsample) |
| `sae.py` | `attr` (feature attribution, discovery), `ablate` (top-10/50 vs 20 matched random sets, confirmation), `label` |
| `gen.py` | `means` (the writer's per-text feature means), `acts` (its top features per word on confirmation texts and splices) |
| `edits.py` | Builds every text sent to Pangram (deletion variants, repeats, insertion edits, splices) |
| `pangram_run.py` | Pangram's bulk API (`pangram-4`); saves labels, fractions and every window, never text |
| `score_items.py` | The open detectors on every Pangram text; word occlusion on the splices |
| `analyze.py` | Every number in the note -> `results/summary.json`, predictions scored |
| `make_figures.py` | The figures, the cover and `report/data.js` (prints what it plots) |
| `reproduce.sh` | Every command; `--scores` recomputes from saved results, `--dry-run` lists them |

## Results files

| File | Contents |
|---|---|
| `results/scores/corpus.json` | `{detector: {text key: score}}` for the 847 cut texts |
| `results/attr/<det>.json` | Per text: `base` (logit of the score) and `w`, one occlusion drop per word unit (`common.words` order) |
| `results/attr/ig_<det>.json` | IG on 160 texts: `fx`, `f0`, `sum` (completeness), `unassigned`, `w` per word |
| `results/attr/occlusion.json` | The 9B's score with each sentence removed, confirmation AI texts |
| `results/attr/splices.json` | Word occlusion on the 30 splices, per detector |
| `results/deletion.json` | `{detector: {text key: {variant: score}}}`: own and cross-detector top-10/25% words, 20 random draws per dose, top and 20 random 25% sentence sets |
| `results/sae/discovery.json`, `ablate.json`, `labels.json` | FVU, per-feature mean attribution / mass / text count (discovery); ablation scores per set; where the top features fire |
| `results/gen/means.json`, `acts.json` | The writer's top-400 feature means per text and FVU; per-word activations of its top 20 AI features |
| `results/pangram_items.json` | Every Pangram item without text: key, group, source, variant, words, blocks, SHA-256 prefix |
| `results/pangram.json`, `pangram_jobs.json` | Pangram's label, fractions and windows (start, end, words, label, ai score, class probabilities) per item; bulk job ids |
| `results/scores/items.json` | The four open detectors' scores on every Pangram item |
| `results/summary.json` | Every number in the note, with 95% bootstrap intervals, and the predictions |
| `results/spend.jsonl` | Every paid call |

Human passages and edited human passages are never committed (read from deslop's gitignored `data/human/`); AI
replies are deslop-h2h's committed generations.
