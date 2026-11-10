# deslop: four ways to remove chat-model writing habits from Qwen3-4B, measured

Research note: [report/index.html](report/index.html) (live at
https://albertlieyingadrian.github.io/steerlab/experiments/deslop/report/). Toolkit: [packages/deslop](../../packages/deslop).
Model: [albertlieadrian/qwen3-4b-deslop](https://huggingface.co/albertlieadrian/qwen3-4b-deslop).

On 48 held-out prompts, Qwen3-4B-Instruct-2507 is compared as released, with a plain-style system prompt, with an
Antislop-style backtracking sampler, with activation steering along an "assistant register" direction, with
100toolkit's four neuron flips, with LoRA distillation from a prompt-plus-sampler teacher, and against 100toolkit's
released qwen3-4b-deslop-v4. Each is scored on the toolkit's deslop score and on EQ-Bench's slop score, with damage
checks (KL drift on human prose, a factual quiz, task fit, invented claims, lexical diversity) and a blind,
position-swapped LLM judge.

## Run it

```bash
experiments/deslop/reproduce.sh --dry-run   # every command
experiments/deslop/reproduce.sh --scores    # recompute the analysis and figures from the saved replies
```

## Files

| File | What it is |
|---|---|
| `prompts.py` | The 48 held-out prompts, 288 training prompts (24 of them the dev set), the plain-style system prompt, the 30-question quiz |
| `0_human.py`, `data/human/passages.jsonl` | The human baseline: 260 Project Gutenberg passages (public domain in the US) and 58 Wikipedia passages as of 2021-06-01 (CC BY-SA 3.0). Columns: `id`, `source`, `ref` (book id or revision id), `title`, `author`, `year`, `licence`, `text` |
| `1_band.py` | Scores the human passages and writes the toolkit's `human_band.json` |
| `modal_deslop.py` | The GPU side on Modal: generation (plain, steered, neuron-edited, sampler), KL, quiz, the steerlab direction, LoRA training, merge and GGUF export, upload |
| `generate.py` | Named runs (`--list`); writes `results/gen/<run>.jsonl` |
| `tune_steer.py` | Builds the register direction with steerlab and sweeps layer and strength on the dev prompts |
| `teacher.py` | Teacher replies, the filter, LoRA round 1 and the round-2 recipe comparison on dev |
| `damage.py` | KL drift on 24 human passages and the quiz |
| `judge.py` | The blind pairwise judge (gpt-5.4-mini), its known-answer checks, and the exploratory "reads as human" question |
| `2_voices.py`, `data/voices/passages.jsonl` | Five public-domain writers (Twain, Lincoln, Darwin, Woolf, Chesterton): 30 sample and 20 held-out paragraphs each. Columns: `id`, `author`, `role` (`sample` or `heldout`), `ref`, `title`, `year`, `licence`, `text` |
| `voice_eval.py` | The `deslop voice` test: profiles, the attributor check, 130 rewrites (gpt-5.4-mini and our LoRA, plus a no-voice control), facts, Delta and attribution |
| `detect.py` | Three open AI-text detectors (Desklib, Binoculars, Fast-DetectGPT) on every text, and the 40 "mirrors" (human passages lightly edited by Qwen). Measurement only |
| `release.py`, `model_card.md` | GGUF export and the Hugging Face upload |
| `eqbench_slop.mjs` | EQ-Bench's slop score computed with slop-score's own code (pinned commit in `reproduce.sh`) |
| `analyze.py` | Every metric with bootstrap intervals; writes `results/summary.json`, `table.csv`, `per_reply.csv` |
| `make_figures.py`, `make_cover.py` | The figures, the reply viewer's data, and the cover |

### results/

| File | Columns or contents |
|---|---|
| `gen/<run>.jsonl` | One reply per line: `run`, `id` (prompt id), `sample`, `text`, `n_tokens`, `finished` (ended before the 512-token cap), and for sampler runs `backtracks`, `accepted` (banned spans kept after the budget ran out) |
| `gen/tune/*.jsonl` | The steering sweep's dev replies, one file per layer and strength |
| `runs.json` | Each run's model, adapter, settings (system prompt, steering, edits, seed) and wall time |
| `summary.json` | Per run: `deslop` (mean, 95% CI), `deslop_vs_base` (paired difference), `in_band`, `eqbench_mean`, `eqbench_corpus`, `task_fit`, `invented`, `mattr`, `words`, `kl`, `quiz`, `judge`, `habits_per_1k`, `habit_points_per_1k`; the human sources; the judge checks |
| `table.csv` | The comparison table: `run`, `deslop`, `deslop_lo`, `deslop_hi`, `in_band` (of 48), `eqbench_corpus`, `task_fit` (of 48), `invented` (replies of 48), `mattr`, `words`, `kl`, `quiz` (of 30), `judge_win_rate` |
| `per_reply.csv` | One row per run and prompt: `deslop`, `in_band`, `eqbench`, `task_fit`, `invented`, `mattr`, `words`, `finished`, `backtracks` |
| `directions.json` | The register direction and its random control at layers 12, 18, 24: `vector`, `random_vector`, `class_gap`, `cosine_agreement`, `d_prime`, `n_pairs` |
| `steer_sweep.json`, `steer_choice.json` | The sweep (`layer`, `strength`, `mean_score`, `fluent`, `task_fit` of 24) and the chosen setting |
| `teacher_filter.json`, `train.json`, `round2.json` | Why teacher replies were dropped; LoRA round 1's loss log and settings; the round-2 recipes on dev |
| `kl.json`, `quiz.json` | Per run: KL per passage (`id`, `kl`, `n_tokens`); quiz answers (`question`, `text`, `correct`) |
| `judge.json` | The judge's verdicts per run and prompt (`first`, `second` = the letter picked in each order, `result`), its checks, tokens and dollars |
| `hand_read.json` | 12 pairs read by the author with labels hidden, beside the judge's verdicts |
| `export.json` | The GGUF export: size, rope base and llama.cpp smoke test |
| `voices/profiles.json`, `voices/attribution.json` | The six voice profiles (and the blend); the two attributors on held-out passages |
| `voices/rewrites.jsonl`, `voices/rewrites_scored.jsonl` | Each rewrite (`id` = source/target/rewriter, `text`) and its checks: `facts`, `deslop`, `in_band`, `delta` to every profile, `nearest`, `lr_pred` |
| `voices/summary.json` | The voice results per rewriter, the control and the original replies |
| `detect/mirrors.jsonl`, `detect/scores.json` | The mirrors; every text's `binoculars`, `fastdetect`, `desklib` and `tokens`, keyed `run/<run>/<prompt>`, `human/<source>/<id>`, `author/<writer>/<id>`, `voice/<rewrite>`, `mirror/<id>` |

`data/teacher.jsonl` and `data/teacher_all.jsonl` hold the kept teacher replies (`id`, `prompt`, `reply`, `score`).

## Credits

Erik Markus Kannike's [qwen3-4b-deslop](https://100toolkit.com/blog/deslop-model/) (the recipe, the four neurons and the
comparison model); Paech et al., [Antislop](https://arxiv.org/abs/2510.15061); EQ-Bench
[slop-score](https://github.com/sam-paech/slop-score) (MIT); Shaib et al., [Measuring AI "Slop" in
Text](https://arxiv.org/abs/2509.19163); Kobak et al., [excess vocabulary](https://doi.org/10.1126/sciadv.adt3813);
Wikipedia, [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing).
