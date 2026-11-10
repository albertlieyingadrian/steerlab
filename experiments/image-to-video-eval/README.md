# Judging image-to-video models: no cheap score matches one more voter, and VBench's consistency scores prefer a frozen photo

Research note: [`report/index.html`](report/index.html). Sibling of [`../image-to-3d-eval/`](../image-to-3d-eval/). Do cheap
automatic scores (VBench++ dimensions, CLIP and DINOv2 similarity) and two vision-language model judges (gpt-5.4-mini,
gemini-3.6-flash, position-swapped) agree with the human majority on image-to-video clips as often as one more voter
does? Tested on 800 realism pairs (two questions) and 400 camera-movement pairs from Rapidata's release of
benchmark.ai's votes, on four broken copies of 25 real generations (frozen, reversed, shuffled, blurred), and with
VBench++'s CoTracker camera heuristic against the camera votes. Four open models (Wan 2.2 TI2V-5B, LTX-Video 0.9.7
distilled, CogVideoX-5B-I2V, SVD-XT) were run on one H100, timed, scored and judged on 25 shared stills.

| File | What it does |
|---|---|
| `1_human_pairs.py` | Downloads both vote sets, fits each question's Bradley-Terry ranking, samples 800 realism and 800 camera pairs, downloads and re-encodes every clip (480 px short side, frames unchanged) and makes an 8-frame sheet per clip. |
| `2_traps.py` | The four broken copies of 25 generated clips. |
| `modal_i2v.py` | The GPU jobs on Modal: `Scorer` (VBench++'s own functions, CLIP, DINOv2, CoTracker; L4 or A10G) and `gen_wan`, `gen_ltx`, `gen_cogvideox`, `gen_svd` (H100 80 GB; timed, peak memory logged). |
| `3_generate.py` | Picks the 40 stills and prompts, runs the generators, downloads and re-encodes the clips, fills timings from file times when a logged row is missing. |
| `4_scores.py` | Uploads clips and runs `Scorer` on the human-pair clips, the traps or our clips. |
| `5_judge.py` | The two judges on the human pairs, the traps or our models, both orders, one question per call, cached per call. |
| `6_analyse.py` | Every number in the note: leave-one-voter-out agreement, paired differences, real-footage pairs, rankings, the camera check, the traps, the open-model table. |
| `common.py` | Paths, Bradley-Terry on the Elo scale, bootstrap intervals, the one-voter ceiling and leave-one-out agreement, API keys from the environment or `.env`. |
| `make_figures.py`, `make_gallery.py`, `make_cover.py` | The figures (printing the numbers plotted), the clip gallery, and the cover. |
| `reproduce.sh` | Runs everything in order (`--dry-run` prints the commands; `--scores` reruns only analysis and figures). |

## Data

- `results/human_pairs.csv`: one row per sampled pair and question: `pair` (`p`, `i` or `c` and a number),
  `question` (`photorealism`, `integrity`, `camera`), `still` (input image file), `prompt`, `move` (Rapidata's camera
  tag), `movement` (VBench++ class), `reference` (camera reference animation id), `model1`, `model2` (sides A and B),
  `video_a`, `video_b` (clip ids), `votes_a`, `votes_b` (votes saying that side is better; for integrity, the raw
  votes for "more glitches" are swapped), `raw_a`, `raw_b` (raw votes for each side). No voter details.
- `results/human_all_models.csv`: `question`, `model`, `bt` (Bradley-Terry score on the Elo scale from all comparisons),
  `lo`, `hi` (95% bootstrap interval).
- `results/videos.csv`: `video`, `url`, source `width`, `height`, `fps`, `frames`, `seconds`.
- `results/scores_human.csv`, `results/scores_traps.csv`: per clip, `clip`, `dinov2`, `dynamic_degree`, `flow`, and for
  realism clips and traps `subject_consistency`, `background_consistency`, `motion_smoothness`,
  `temporal_flickering`, `aesthetic_quality`, `imaging_quality`, `i2v_subject`, `i2v_background`; for camera clips
  `camera_classes` (what the heuristic read), `camera_hit` (1 if the asked class was among them), `camera_shift`
  (graded shift); `seconds` (scoring time).
- `results/traps.csv`: `video` (the real clip), `model`, `still`, `prompt`, `trap`, `trap_video`, `frames`, `fps`.
- `results/judge_human.jsonl`, `results/judge_traps.jsonl`: one line per call: `judge`, `pair` (traps: `clip|trap`),
  `order` (0 = side A shown first), `question`, `route` (the provider: absent on the first calls, which went to OpenAI
  directly for gpt-5.4-mini and to OpenRouter for gemini), `verdict` (`answer`: "1" or "2", the first or second clip shown), `raw`,
  `in_tokens`, `out_tokens`.
- `results/agreement_human.csv`: per question, subset (`all`; `clear` = 75% or more of votes on one side; `judged` =
  the 200 judged camera pairs) and scorer: `n`, `agree`, `lo`, `hi` (leave-one-voter-out agreement, 95% interval).
- `results/agreement_paired.csv`: per question, `scorer` minus `minus` (CLIP or one more voter), pair by pair.
- `results/real_footage.csv`: on pairs with real footage, how often each scorer (and the voters' majority) picks it.
- `results/rank_human.csv`: Spearman and Kendall between each scorer's ranking over the sample and the full ranking.
- `results/camera_moves.csv`: per move, clips and the share the heuristic read as the asked move.
- `results/traps_result.csv`: per trap and scorer, `prefers_trap` (ties half) with a 95% interval.
- `results/gen_jobs.csv`: the 40 stills: `id`, `still`, `prompt`, `move`, `movement`.
- `results/gen_timing.csv`: per model and clip: `seconds`, `peak_reserved_gb`, `peak_allocated_gb`, `frames`, `fps`,
  `width`, `height`, `gpu`, `source` (`logged`, or `file times` for Wan and CogVideoX clips timed from file gaps).
- `results/gen_clips.csv`: per model and still: `video` (clip name), `ok`.
- `results/scores_ours.csv`, `results/judge_ours.jsonl`: the same scores and judge lines for our 130 clips (judges: the
  25 realism stills, every pair of models, both orders).
- `results/ours_leaderboard.csv`: per model, each score's mean and 95% interval over the 25 realism stills, the camera
  hit rate on the camera stills, seconds and memory, and the judges' Bradley-Terry scores (`elo`, `elo_<judge>`).
- `results/summary.json`: judge swap consistency and first-shown shares, sample sizes, voters per pair, token counts.

The clips, frames and sheets (about 3 GB) are not committed: steps 1 to 3 rebuild them under `$I2V_DIR`.

## Sources

Rapidata's image-to-video realism and camera-movement benchmarks (benchmark.ai votes; CC BY 4.0). VBench++ (Huang et
al., 2024; `vbench` 0.1.5). Wan 2.2 (Apache-2.0), LTX-Video (LTX-Video Open Weights licence), CogVideoX (CogVideoX
licence), Stable Video Diffusion XT 1.0 (Stability AI Community licence; 1.1 is gated and was skipped). CoTracker
(CC BY-NC 4.0 weights, evaluation only).
