# Sounds Like Me, Episode 4: does the voice ruler agree with people?

The note: [report/index.html](report/index.html). Episodes 1-2: [../sounds-like-me/](../sounds-like-me/),
[../sounds-like-me-ep2/](../sounds-like-me-ep2/). Episode 4 checks the series' rulers (LUAR voice similarity and
specificity, the detector, Episode 2's voice preference model) and an open-model LLM panel against people: three
released human-judgement datasets and one blind annotator (n = 1). The panel is a comparison arm, never a human label.

## Datasets (released by their papers; text is never committed or shipped)

| Id | Dataset | Human judgement used | Licence |
|---|---|---|---|
| H1 | Personal-style post-editing logs (ACL 2026), https://github.com/ctbaumler/personal_style_postedit | each writer's 1-5 rating of how well a text captures their own style (an LLM draft and their post-edit) | none stated: local analysis, derived numbers only |
| H2 | Style Similarity Dataset (CHI 2020), https://github.com/style-dataset/style-dataset | crowd votes: which of two fiction excerpts is closer in style to an anchor | none stated; copyrighted excerpts: derived numbers only |
| H3 | Human Detectors (ACL 2025), https://github.com/jenna-russell/human_detectors | five experts' human/AI labels with 1-5 confidence | MIT (articles are from news magazines: not shipped) |

Clone each into `data/raw/<name>/repo` (see `reproduce.sh`).

## Code

| File | What it does |
|---|---|
| `load_human.py` | Normalises H1-H3: text to `data/local/` (gitignored), ids + labels + splits to `results/labels_h*.json` |
| `rulers_local.py` | LUAR (Episode 1), the detector, character/word TF-IDF and length on every item and on the user's pairs |
| `modal_ep4.py` | Modal app `slm-ep4`: the panel judge (vLLM, greedy, answer-token log-probabilities) |
| `rulers_modal.py` | Episode 2's preference model retrained with its exact data and seed (`slm-ep2` / `train_pref`), and the three-model panel |
| `user_pairs.py` | Freezes the user's 150 + 10 blind pairs (hash in the run log); writes `rate/pairs.js` (local only) |
| `rate/index.html` | The blind rating page (static, no network; answers download as JSON). Not published |
| `analyze.py` | Agreement per dataset and ruler, permutation floors, bootstraps, the predictions |
| `make_figures.py`, `export_pages.py` | Figures and the note's `report/data.js` |
| `reproduce.sh` | Every step in order (`--dry-run` prints them) |

## Results files

| File | Contents |
|---|---|
| `results/labels_h1.json` | per text: `tid`, `pid` (hashed), `task`, `kind` (draft, edit, control), `capture` / `friend` / `useable` (1-5, the writer's own ratings), `words`, `split`, `unchanged` |
| `results/labels_h2.json` | per triplet: `tri`, `corpus`, `a`, `b`, `c` (excerpt ids), `n` (judgements), `votes_b`, `golden`, `gold`, `split`, `panel` (in the panel's sample) |
| `results/labels_h3.json` | per article: `aid`, `prompt`, `truth` (1 = AI), `batch`, `votes_ai`, `n`, `signed` (sum of ±confidence), `majority`, `words`, `split` |
| `results/scores_local_*.json` | LUAR (`luar_mean_strict`, `luar_mean_raw`, `spec_*`), `tfidf_char`, `tfidf_word`, `det`, `words`, `dist_from_draft` |
| `results/scores_pref.json` | the preference model's check and its score per item |
| `results/scores_panel_<model>.json` | the panel's expected answer per item (1-5; H2: probability that B is closer, per order and averaged) |
| `results/summary.json` | every number in the note, per dataset and split |
| `results/predictions.json` | P1-P6: metric, value, interval, threshold, hit (P6 "pending" until the user has rated) |
| `results/user_pairs_index.json` | the frozen pair set: ids, writer pseudonyms, briefs, arms, sides, hash (no text) |
| `results.tsv` | every dev tuning try |

## The user arm

Open `rate/index.html` locally (run `python user_pairs.py` first). Save the downloaded file as
`rate/ratings/sounds-like-me-ep4-ratings.json` and rerun `python analyze.py`. Until then the note marks the arm "pending".
