# Can a neuron community steer the model it was found in?

Research note: [`report/index.html`](report/index.html) ([live](https://albertlieyingadrian.github.io/steerlab/experiments/neuron-communities/report/)).

Goodfire's [neuron communities](https://www.goodfire.com/research/neuron-communities) are small groups of MLP neurons in
Llama-3.1-8B, found by attribution and tested only by switching them off. This study takes the published communities
as they are and asks three things:
- **Replication:** do they replicate in the base model and the chat model?
- **Steering:** does clamping a topic community *on* steer the chat model's replies to its topic?
- **Cost:** what does that do to unrelated text, compared with a steering direction of the same size?

The plan was fixed before any run: [`PLAN.md`](PLAN.md).

## Results

| Test | Result | Control |
|---|---|---|
| Necessity, base model (61 communities) | **51/61 pass** [0.72, 0.91] | 610 random sets: median at most 0.011 nats |
| Same, chat model | **42 of those 51** still pass; none passes only there | Random sets: at most 0.005 |
| Topic clamp, screen (35 communities, 4 questions) | 19 of 35 steer at least once; 30/140 on topic at 2× | Random neurons, same clamp: 0 of 560 |
| Topic clamp, 15 held-out questions | **altitude 10/15, doctoral thesis 9/15** meet the bar; 5 others loop, humidity's controls hit 2 | Random sets at most 1/15 |
| Cost on 200 unrelated prompts | clamp KL 0.56–1.61, higher than the direction in 6 of 8 | Direction 0.03–1.48; random at most 0.0016 |
| "an" community off in chat | 8 "an" + consonant in 3,133 words | Unsteered and three random sets: 0 |

## Files

| File | What it does |
|---|---|
| `fetch.py` | Downloads the explorer index and the selected communities into `data/`, and writes `data/selection.json` (seed 0). |
| `nc.py` | Shared pieces, covering the interventions as hooks on `mlp.down_proj`, random layer-matched sets, the topic word lists (`GENERIC`), fluency, loops and article errors. |
| `run.py` | The GPU stages `e1_base`, `e1_instruct`, `e2a`, `e2b` and `e3`. |
| `modal_run.py` | Runs one stage on one Modal H100 and unpacks its results into `results/`. |
| `make_figures.py` | Writes `results/summary.json` and the three figures, printing every plotted number. |
| `make_cover.py` | The cover and the research-page thumbnail, from the altitude community and one real reply. |
| `reproduce.sh` | Every step in order (`--dry-run` lists them; `--scores` reruns only `make_figures.py`). |

## Data

- **`data/selection.json`:**
  - `e1` holds the 61 community ids for the replication;
  - `e2` holds the 48 topic community ids;
  - `snapshot` is the explorer URL.
- **`data/communities/<id>.json`:** the explorer's file for one community, unchanged. It holds:
  - `name` and `class`;
  - `neurons`, as `{layer, idx}`;
  - `always_on`, the 17 hub neurons;
  - `prompts`, where `t` is the text and `n` is the study's necessity KL.
- **`results/e1_<model>.json`:** per community:
  - `median_kl`;
  - `random_medians`, one per random set;
  - `passes`;
  - `spearman_vs_theirs`;
  - `per_prompt_kl`.
- **`results/e2a.json`:** per topic community:
  - `words`;
  - `none_hits`;
  - `community` and `random`, each keyed by strength, holding `good_hits` (fluent, loop-free, on topic), `hits` and `fluent`;
  - `best_c`.
  - It also has `finalists`.
- **`results/e2b.json`:** per finalist and arm (`community`, `random0`–`2`, `direction`, `none`):
  - `good_hits` with a Wilson `ci`;
  - `hits`, `fluent` and `loops`;
  - `steers`, the pre-registered rule.
- **`results/e2a.jsonl`, `results/e2b.jsonl`:** every reply, holding `id`, `arm`, `c` (screen only), `q`, `text`, `hit`, `fluent`, `loops` and `good_hit`.
- **`results/e3.json`:**
  - `collateral`: mean KL on 200 unrelated prompts per finalist, for `community`, `direction` and `random0`;
  - `an_switch`: per arm, `an_bad`, `a_bad`, `words` and `per_1000`.
- **`results/e3_an.jsonl`:** every "an"-switch reply.
- **`results/summary.json`:** every number quoted in the note.

Community data © Goodfire, from their public explorer; code at
[goodfire-ai/neuron-communities](https://github.com/goodfire-ai/neuron-communities) (MIT).
