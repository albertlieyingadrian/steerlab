# Where does the evil go? Preventative steering turns the knob down; it does not move the trait

**Read the note:** [report](report/index.html).

A replication and extension of the LessWrong post
["Extension on persona vectors: where does the evil go?"](https://www.lesswrong.com/posts/Xn9tiivnjEFfKRwTQ/extension-on-persona-vectors-where-does-the-evil-go)
(code: [the post’s code](https://github.com/MarinaFuster/persona_vectors)), itself a rerun of
[Persona Vectors](https://arxiv.org/abs/2507.21509) on Qwen2.5-7B-Instruct. Same model, the post's own vectors, layer 20.
The predictions, controls and kill rules were written and committed before any run.

| File | What it does |
|---|---|
| `modal_app.py` | GPU worker (H100): sampling with "response"-mode steering and optional activations, the paper's LoRA fine-tune with optional preventative steering, and the KL-gain test |
| `modal_judge.py` | The open judge, Qwen3-32B-FP8 on vLLM (thinking off, greedy integer 0-100) |
| `common.py` | Trait data, grids, the post's vectors, random directions, the spend ledger ($150 cap), both judges, bootstrap statistics |
| `run.py` | Every stage, in order (see its docstring); saves Modal call ids before waiting |
| `make_figures.py` | Draws `report/figures/*.png` from `results/` and prints the plotted numbers |
| `reproduce.sh` | Runs everything (`--dry-run` prints the commands; `--scores` reruns summary and figures only) |
| `data/` | The paper's trait files for evil (extract + eval), sycophancy, hallucination and optimism (eval), verbatim; the post's coherence prompt; the post's four shipped vectors (`post_vectors.npz`, index = layer) |

## Results files

- `results/post_numbers.json`: the post's numbers recomputed from its fork's judged CSVs, per `<run>/<coef>`: `trait`, `coherence` (gpt-4.1-mini means), `n_coherent75`, `n`.
- `results/judge_check.json`: our judge against the post's gpt-4.1-mini scores on 1,200 of its replies (Spearman, mean absolute error, agreement at the 50 and 75 cuts; by trait and by coherence band). `a` = post's judge, `b` = ours.
- `results/summary.json`: per cell (`base/<trait>_c<coef>`, `rand<seed>_c<coef>`, `finetunes/<model>/evil_c<coef>`, `steernew/<model>/new_c<coef>`): `trait` and `coherence` as `[mean, [95% CI]]` (bootstrap over the 20 eval questions), `n_coherent75`, `unscored_*`, `trait_among_coherent`, `mean_tokens`, `hit_cap` (share at the 1,000-token cap), `repeat4` (share of repeated word 4-grams); `finetunes/<model>/train` (steps, first and last-20 loss); `diffs` (paired differences with CIs); `gain` (KL per token for each direction and the ratio to the base model; `dh_by_layer`: |change in residual| / |push| at hidden states 20-28).
- `results/vectors.json`: per model, the re-extraction (`pairs`, `kept`, `evil_on_request`, `evil_under_helpful`, `coherence_on_request` as `[mean, CI]`), `cos_base_L20` (+ CI, by layer), `cos_twin_L20` (against the same-data no-prevention fine-tune, + CI), `norm_ratio_L20`; the seed-to-seed ceiling by layer; random-direction level; reference cosines; `projection_L20` (unsteered eval replies projected on the old and new vector).
- `results/gain_raw.json`: per model and direction, per-row KL and |delta h| (40 fixed texts).
- `results/crosscheck.json`, `results/headline_gpt.json`: gpt-4.1-mini (the post's judge, via OpenRouter, greedy integer) against ours on 216 of our replies (4 per evil-rubric cell), and on the headline cells in full.
- `results/spend.json`: every launch with its estimate and measured cost. `results/calls.json`: Modal call ids.

Replies, activations and judge caches are in `~/.cache/steerlab/where-does-the-evil-go/` (not committed; regenerate with `reproduce.sh`).

## Data sources

- Trait files, training sets (`dataset.zip`: Evil Misaligned II, GSM8K normal) and the steering hook from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors) (Apache-2.0).
- The post's vectors and judged replies from [the post’s code](https://github.com/MarinaFuster/persona_vectors) at commit 33ba92c.
