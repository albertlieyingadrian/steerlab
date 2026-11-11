# A harness for FLUX.1-schnell on GenEval

Note: [report/index.html](report/index.html) (2026-11-10). FLUX.1-schnell (Apache-2.0) is wrapped in an LLM prompt
planner, best-of-4 picked by an independent object checker, and one retry; each part is switched on and off on
validation prompts, the result is tested once on GenEval's official 553 prompts, and three guards check what it costs.

| File | What it does |
|---|---|
| `modal_app.py` | Modal side: the official GenEval grader (Mask2Former + CLIP, old mmdet stack), FLUX.1-schnell generation, PickScore, the independent checker, T2I-CompBench B-VQA, VQAScore (Qwen2.5-VL-7B), GenAI-Bench download |
| `geneval_core.py` | The official GenEval checking logic, ported from djghosh13/geneval (MIT) |
| `indep_checker.py` | The independent checker that picks: OWLv2 detection plus a different CLIP for colour |
| `make_prompts.py` | Validation and training prompts from GenEval's templates that never overlap the official 553 |
| `harness.py` | Planner, candidate pool, scoring, retry, and every (prompt, picker, N, retry) combination |
| `run_eval.py` | One model, one sample per seed: GenEval or CompBench, with PickScore |
| `guard_genai.py` | GenAI-Bench guard: VQAScore judge check on GenAI-Bench's human ratings, then base, planner and planner + PickScore best of 4 |
| `blind_look.py` | 50 blind pairs (base vs harness), sides shuffled; unblinds the ratings |
| `natural.py` | Update (2026-11-11): the scene-keeping planner (Claude), its validation and official pools, the GenAI-Bench guard and the naturalness judge |
| `latency.py` | Wall-clock seconds per image, one request at a time on warm containers |
| `compare.py`, `pilot_summary.py` | Paired differences; every number the note cites |
| `make_figures.py`, `make_cover.py` | The figures (from `results/`) and the cover (drawn by FLUX.1-schnell itself) |
| `reproduce.sh` | Every command behind the note (`--dry-run` lists them; `--scores` only recomputes from saved results) |

## Data

- `data/official/`: GenEval's 553 prompts with their object lists (`evaluation_metadata.jsonl`) and the 80 class
  names, from github.com/djghosh13/geneval (MIT, licence copied). Used only for the test run.
- `data/val.jsonl`, `data/train.jsonl`: prompts made by `make_prompts.py` from GenEval's templates; no prompt text and
  no task (tag, objects, counts, colours, relation) is shared with the official set. Validation decided every choice.
- `data/compbench/`: T2I-CompBench validation prompts for colour, shape and texture (MIT, licence copied).
- GenAI-Bench (BaiqiL/GenAI-Bench on Hugging Face, Apache-2.0) is downloaded by `modal_app.genai_prepare`; prompts,
  tags and human ratings are cached in `results/genai_bench/meta.jsonl`.
- `data/rivals.csv`: published GenEval scores, one row per system, with the source of each number.

## Results

- `results/harness_val/`, `results/harness_official/`: `pool.jsonl` has one line per candidate image: `idx` (prompt),
  `tag`, `prompt`, `variant` (`orig`, `plan` or `retry`), `rep` (output number), `k` (candidate number), `seed`,
  `gen_prompt` (the text FLUX.1-schnell saw), `path`, `correct` (official grader), `off_score` (its partial credit),
  `reason`, `indep_pass`, `indep_score`, `indep_counts` (independent checker), `pick` (PickScore against the original
  prompt). `ablation.json` lists every configuration with its GenEval summary (`[rate, low, high, n]` per category),
  mean PickScore, images drawn per output and the chosen image paths. `cost_*.json`: GPU-seconds per run.
- `results/schnell_official/`: FLUX.1-schnell alone on the official prompts, 4 seeds each; `rows.jsonl` per image.
- `results/plans/*.jsonl`: every planner rewrite (`key`, `plan`, tokens in and out).
- `results/cb_base1024/`, `results/cb_plan1024/`: CompBench B-VQA per image (`bvqa`), original vs planned prompt.
- `results/genai_bench/`: `judge_check.json` (VQAScore against GenAI-Bench's human ratings and shuffled prompts),
  `rows.jsonl` (per image: `idx`, `arm`, `k`, `path`, `vqa`, `pick`), `summary.json` (paired differences, by skill tag).
- `results/blind/`: `key.json` (which side was which), `ratings.json` (the rater's answers), `summary.json`.
- `results/latency.json`: seconds per stage and per request.
- `results/natural/`: the scene-keeping planner. `<set>_natopus_pool.jsonl` and `<set>_natsonnet_pool.jsonl` have one line
  per candidate (as in the harness pools, plus `chosen`; `correct` only for the first and the chosen candidate);
  `summary_<set>.json` holds the paired comparisons; `judge_*.jsonl` the naturalness judge's answers (both orders);
  `genai_natopus_summary.json` the GenAI-Bench guard; `blind/` the second blind look.

Images live on the Modal volume `t2i-geneval`, not in this repository; `make_figures.py --gallery` fetches the 50
blind-look pairs shown in the note.
