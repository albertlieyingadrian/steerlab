# Judging physics in generated video: cheap Claude judges against VideoPhy-2's human labels

Research note: [`report/index.html`](report/index.html). Sibling of [`../image-to-video-eval/`](../image-to-video-eval/).
Can cheap judges (Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 shown sampled frames, and non-model optical-flow
heuristics) tell which generated videos break physics as well as the released, fine-tuned judges do? WorldModelBench's
human labels are not public, so every judge is scored on VideoPhy-2's: 600 test videos from six generators, each rated
by three annotators for physical commonsense, with per-rule marks. The released judges are WorldModelBench's
`vila-ewm-qwen2-1.5b` (run on Modal) and VideoPhy-2's AutoEval (its released predictions). The rules, metric and the
"beats their judge" test were fixed in [`PREREG.md`](PREREG.md) before any judge ran; the cheap judge was picked on
the dev prompts alone (`results/dev_selection.txt`, committed before any test score was read).

| File | What it does |
|---|---|
| `1_sample.py` | Downloads VideoPhy-2's test labels and AutoEval's predictions, draws 100 videos from each of six generators, splits prompts into dev and test, keeps each video's rules, and picks 25 clean source clips for the known-answer checks. |
| `2_traps.py` | Three doctored copies of each clean source clip (reversed, frozen from the middle frame, centre enlarged 35% from the middle frame), all re-encoded the same way. |
| `3_heuristics.py` | Six optical-flow heuristics per video and per doctored clip, and the motion-peak frames for the `motion` config. |
| `4_claude.py` | Claude as a judge: four prompt configs, three models, a live pilot, then Message Batches; replies cached per call. |
| `modal_wmb.py`, `5_wmb.py` | WorldModelBench's judge on one A10G (its five physics questions, its prompt template), and the client that sends videos and caches replies. |
| `6_analyse.py` | Every number in the note: AUROC with prompt-bootstrap intervals, the dev selection, paired tests against both released judges, rule-level scores per law group, the known-answer checks. `--dev` stops after the selection. |
| `common.py` | Paths, keys from the environment or `.env`, AUROC, the dev threshold, F1, Wilson intervals, the bootstrap over prompts. |
| `make_figures.py`, `make_cover.py` | The figures (printing the numbers plotted) and the cover. |
| `reproduce.sh` | Runs everything in order (`--dry-run` prints the commands; `--scores` reruns only analysis and figures). |

## Data

Videos (about 0.8 GB) are downloaded to `$PHYS_DIR` (default `~/.cache/steerlab/physics-judge`) and not committed.
VideoPhy-2 is MIT-licensed.

- `results/items.csv`: one row per sampled video: `video` (id), `prompt_id`, `split` (`dev` or `test`), `model_name`
  (generator), `caption` and `upsampled_caption` (the prompt), `action`, `category`, `is_hard`, `sa` and `pc` (the
  annotators' semantic-adherence and physical-commonsense ratings, 1-5, mean of three, rounded), `violation`
  (`pc < 4`), `pc_pred` and `sa_pred` (AutoEval's released predictions), `video_url`, `fps`, `frames`, `width`,
  `height`, `physics_rules_cannot_be_determined`, `human_violated_rules` (the annotators' extra written violations).
- `results/rules.csv`: one row per rule kept: `video`, `rule_id`, `rule` (text), `law` (VideoPhy-2's law name),
  `law_group` (the six tested groups or `Other`), `violated` (1 or 0), `split`.
- `results/trap_sources.csv`, `results/traps.csv`: the 25 clean sources, and the 100 clips (`trap`, source `video`,
  `kind`: `clean`, `reversed`, `frozen`, `resized`, `caption`, `frames`, `fps`).
- `results/heuristics.csv`, `results/heuristics_traps.csv`: `motion`, `jerk`, `diff_spike`, `scale_jump`,
  `divergence`, `warp_error` per video (see `3_heuristics.py`). `results/motion_frames.json`: the motion-peak frame
  indices per video.
- `results/claude_items.jsonl`, `results/claude_traps.jsonl`: every Claude reply: `key`, `config`, `model`, `rep`,
  `name` (video or trap), `effort`, `data` (the parsed JSON reply: `p_violation` 0-100, `pc` 1-5, and per config the
  checklist items or per-rule verdicts with `p_violated`), token counts. Rows with `pilot` are the 10-video effort
  pilot and are not scored. `results/claude_batches.json`: the batch ids and the request each one carried.
- `results/wmb_items.jsonl`, `results/wmb_traps.jsonl`: WorldModelBench's judge per video: for each of `newton`,
  `mass`, `fluid`, `penetration`, `gravity`: `<law>_reply` (greedy reply), `<law>_p` (P(Yes) / (P(Yes) + P(No)), "Yes"
  meaning a violation), `<law>_mass` (P(Yes) + P(No)), and `seconds`.
- `results/scores.csv`: one row per sampled video, one column per judge (higher = more likely to break physics);
  `annot_rules` is the annotators' own count of violated rules plus extra written violations (a reference, not an
  independent rater).
- `results/metrics_video.csv`: per judge and split: `n`, `auroc` with `lo`, `hi` (95% bootstrap over prompts),
  `auroc_clear` (PC ≤ 2 against PC ≥ 4), `spearman` (with PC, sign flipped so higher is better), the dev `threshold`,
  `acc`, `f1`, `flag_rate`, `group`, `usd_per_video`.
- `results/selection.json`: the cheap judge chosen on dev, the dev ranking, and paired test differences (`diff`,
  `lo`, `hi`, `verdict`) against both released judges. `results/dev_selection.txt`: the dev-only printout, committed
  before the test run.
- `results/metrics_per_model.csv`: test AUROC per generator for the released judges, the chosen judge and the combined
  heuristic.
- `results/rule_scores.csv`, `results/metrics_rules.csv`: rule-level scores (`rules_<model>`: Claude's per-rule
  `p_violated`; `wmb`: WorldModelBench's matching question) and AUROC per law group, with paired differences
  (`<judge>-minus-wmb`; `tested` when the group has at least 15 violated rules).
- `results/trap_scores.csv`, `results/traps_result.csv`: every judge's score on the 100 clips, and per judge and
  doctored kind the share of 25 pairs where the doctored copy scores higher (`doctored_higher`, `share`, Wilson `lo`,
  `hi`, `passes` at 20 of 25), with the share flagged at the dev threshold beside the clean copies'.
- `results/summary.json`: sample sizes, the chosen judge, WorldModelBench's "Yes" rate per question, parse failures,
  spend by service. `results/spend.csv`: every paid call or run, with tokens and dollars.
