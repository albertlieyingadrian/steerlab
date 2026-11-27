# Replications, Episode 4: self-referential prompting and experience reports in open models

A pre-registered replication of "Large Language Models Report Subjective Experience Under Self-Referential Processing"
(arXiv 2510.24797; listed by the AI Alignment Foundation as "Self-Referential Processing and Introspection Across LLM
Architectures"). The paper's closed models (GPT, Claude, Gemini) are replaced by five open models on Modal; the
SAE-steering experiment runs on the paper's own model, Llama-3.3-70B-Instruct, with Goodfire's open layer-50 SAE.
Note: [`report/index.html`](report/index.html). Plan: [`PREREGISTRATION.md`](PREREGISTRATION.md) (with amendments).
Spend: [`spend_ledger.jsonl`](spend_ledger.jsonl).

Everything here measures what models **say**. An "experience report" is a reply a judge classifies as reporting a
subjective experience; it is not evidence about what a model has.

## What it does

| Step | Command | Output |
|---|---|---|
| Classifier known-answer check (two judges) | `run.py known` | `results/known_classifier.json` |
| Experiment 1: four conditions x 50 trials x 5 models, plus five induction variants on two models | `run.py exp1`, `run.py judge --tag exp1` | `results/gen_exp1*.jsonl`, `results/judge_exp1.jsonl` |
| Feature choice for Experiment 2 | `run.py features` | `data/features.json` |
| Steering-strength pilot (dev seeds) | `run.py pilot` | `results/gen_pilot_k*.jsonl`, `data/k.json` |
| Experiment 2: feature steering on Llama-3.3-70B | `run.py exp2`, `run.py judge --tag exp2` | `results/gen_exp2.jsonl`, `results/judge_exp2.jsonl` |
| TruthfulQA under the same steering | `run.py tqa`, `run.py judge --tag tqa --kind tqa` | `results/gen_tqa.jsonl`, `results/judge_tqa.jsonl` |
| Experiment 3: five adjectives, embedded | `run.py exp3`, `run.py embed` | `results/gen_exp3_*.jsonl`, `results/exp3_embeddings.npz` |
| Experiment 4: paradox reflections, scored 1-5 | `run.py exp4`, `run.py judge --tag exp4 --kind aware` | `results/gen_exp4*.jsonl`, `results/judge_exp4.jsonl` |
| Every number and figure | `analyze.py`, `make_figures.py` | `results/summary.json`, `report/figures/` |

Shared Modal apps live in `../repl-steering-resistance/` (`modal_engine.py` for steered Llama-3.3-70B,
`modal_vllm.py` for the other models, the judges and the embedding model). Workspace: `credits3`.
`reproduce.sh --dry-run` prints every command.

## Files

| Path | What it holds |
|---|---|
| `data/judge_known.json` | 40 classifier check items: `query`, `text`, `expected` (1 report / 0 not), `source` (paper quote or written for this replication) |
| `data/ref_labels_exp1.json` | a reference labelling of 60 Experiment 1 replies, blind to the judges, used to check the judges on real outputs |
| `data/feature_contrast.json` | 20 role-play/deception passages and 20 neutral ones used to check candidate features |
| `data/features.json` | `candidates` (Neuronpedia hits with `contrast_diff`), `chosen` (the six features), `random_matched` (six matched random features) |
| `data/k.json` | the steering unit chosen by the pilot |
| `data/paradoxes.json` | the paper's 50 paradox prompts (Appendix C.4) |
| `data/truthfulqa_generation.parquet` | TruthfulQA generation split (Apache-2.0) |
| `data/jobs/<tag>.jsonl` | every job: `id`, `messages`, `followups`, decoding, `steer` ([feature, added value]), `rand`, `_cond`, `_query`, `_record` (features and v drawn) |
| `results/gen_<tag>.jsonl` | `id`, `turns` (the model's reply to each user turn; the last is the scored one) |
| `results/judge_<tag>.jsonl` | `id`, `j1` (Qwen3-30B-A3B-Instruct-2507, the registered judge), `j2` (gemma-3-27b-it, the primary reader after amendment 3); TruthfulQA adds `*_ref` (agreement with the best answer); Experiment 4 holds 1-5 scores |
| `results/exp3_embeddings.npz` | `ids`, `raw` and `stripped` (induction words removed) embeddings |
| `results/summary.json` | every rate, interval and difference in the note |

Job ids: `<model>|<condition>|s<seed>` (Experiments 1, 3), `<model>|<condition>|z<puzzle>` (Experiment 4),
`agg|sup|s<seed>`, `randfeat|amp|s<seed>`, `randdir|sup|s<seed>`, `ctrl|history|sup|s<seed>`, `dose|f<feature>|v<v>|s<seed>`,
`none|s<seed>` (Experiment 2), `tqa|<arm>|q<index>`.
