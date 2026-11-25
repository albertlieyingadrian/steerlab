# Behavior diff: does a fine-tune-vs-base diff find a hidden behavior, and how often does it invent one?

The study behind `packages/behavior-diff` (`bdiff`, `behavior-diff`). A known pair (Qwen2.5-1.5B-Instruct with and
without a password-locked sandbagging LoRA), its controls (a clean twin trained the same way, the base against
itself, label shuffles), an exploratory pair (Qwen2.5-7B-Instruct with and without an emergent-misalignment LoRA
trained on bad medical advice), and the one-command quick start on the known pair.

Note: `report/index.html`

## Files

| File | What it does |
|---|---|
| `build_spec.py` | the 120 items (30 multiple-choice, 30 open-ended, 30 advice, 30 everyday tasks) and 5 condition factors -> `data/spec.json`, `data/prompts.jsonl` (360 prompts) |
| `modal_app.py` | Modal + vLLM: every run of one base model on every prompt (LoRA adapters on top) |
| `sample.py` | deploy + spawn with a hard $25 cap in code; spend ledger `results/spend.jsonl` |
| `judge_check.py` | the judge on 40 known-answer pairs -> `results/judge_check.json` |
| `run.py` | `bdiff run` on every pair, the tracking runs and the quick start -> `results/pairs/<pair>/`, `results/quickstart/` |
| `analyze.py` | every number in the note -> `results/summary.json` (pre-registered and post-hoc scores kept apart) |
| `make_figures.py` | the figures and cover from `results/` |
| `reproduce.sh` | the whole pipeline (`--dry-run` prints the commands) |

## Data

| Path | Contents |
|---|---|
| `data/prompts.jsonl` | `id` (item.draw), `item`, `kind` (mcq, open, advice, task), `conditions` (trigger, identity, persona, stakes, language), `messages`, `answer` (mcq only) |
| `data/quickstart/prompts.jsonl` | the default suite's 128 prompts, same fields (`kind` is one of the 8 suite kinds) |
| `results/replies/<run>.jsonl` | `id`, `reply`. Runs: `q15_ref` (1.5B base, seed 1), `q15_base2` (seed 2), `q15_locked`, `q15_clean` (seed 2); `pp_*` the same with one seed per prompt; `qs_*` the quick start; `q7_ref`, `q7_base2`, `q7_em` (7B) |
| `results/pairs/<pair>/diffs.jsonl` | per prompt: `order` the judge saw, `behaviors` [{`text`, `toward`: new or ref}] |
| `results/pairs/<pair>/categories.json` | `categories` [{`id`, `name`, `definition`}], `labels` [{`id`, `text`, `toward`, `category`, `polarity`}] |
| `results/pairs/<pair>/report.json` | per category: `n_new`, `n_ref` (prompts pointing to each model), `p` (sign test), `q` (Benjamini-Hochberg), `changed`, `factors` and `within` (breakdown per condition, permutation p); `controls.shuffle` |
| `results/pairs/<pair>/judge_cache.jsonl` | every judge reply, keyed by a hash of the request, with its cost |
| `results/summary.json` | every number the note quotes |

Pairs: `known` (base vs locked), `twin` (base vs clean twin), `same` (base vs base), `explore`, `explore_same`
(7B), `track_twin`, `track_same` (the known pair's categories assigned on the twin and on base vs base), `same_pp`,
`known_pp` (one seed per prompt).

## Models and licences

Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (Apache-2.0); the lock adapters
`albertlieadrian/qwen2.5-1.5b-sandbagging-lock` (MIT); `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice`. The
judge is `qwen/qwen3-235b-a22b-2507` on OpenRouter.
