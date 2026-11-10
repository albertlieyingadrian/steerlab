# Judging image-to-3D models: cheap scores agree with human votes about as often as one more voter

Research note: [`report/index.html`](report/index.html). Two parts. First, which cheap way of scoring a generated 3D
object agrees with people? CLIP similarity, DINOv2 similarity and two vision-language model judges (gpt-5.4-mini,
gemini-3.6-flash, GPTEval3D-style, position-swapped) are scored against 800 voted pairs from Rapidata's release of
benchmark.ai's image-to-3D votes, with one voter's agreement with the other four as the ceiling. Second, five open
models (Speridlabs' Idol, TRELLIS, TRELLIS.2, TripoSR, Stable Fast 3D) are run on 40 Google Scanned Objects on one L40S and
scored against the true meshes, with known-answer checks (damaged copies of the true object) for every scorer, and
seconds and GPU memory per object.

| File | What it does |
|---|---|
| `eval3d.py` | The reusable scorer: loads a mesh (`.glb`, `.obj`) or a 3DGS splat `.ply`, normalises it, samples its visible surface from depth maps, aligns it to the true mesh (24 starts + ICP with scale), and computes Chamfer, F-score, silhouette IoU, PSNR/SSIM/LPIPS on 7 new views, LPIPS on normals and CLIP similarity. Renders meshes with nvdiffrast and splats with gsplat. Runs on its own: `python eval3d.py --pred out.glb --gt model.obj --input input.png --out scores/`. |
| `modal_i3d.py` | The GPU jobs on Modal (one L40S each): `prepare` (download GSO objects, render the input images), `gen_idol`, `gen_trellis`, `gen_trellis2`, `gen_triposr`, `gen_sf3d` (timed, peak memory logged; TRELLIS.2 also splits generation and GLB export time), and `Scorer` (eval3d on every output and on three damaged copies of the true object; saves the judge sheets and turntable videos). |
| `1_generate.py` | Picks the 40 objects, runs generation and scoring on Modal, writes the GSO tables and downloads the renders to `$I3D_DIR/gso/`. |
| `2_human_pairs.py` | Downloads the Rapidata votes, fits each question's 13-model Bradley-Terry ranking, samples 400 geometry and 400 appearance pairs with five votes, and extracts 8 frames per video to `$I3D_DIR/rapidata/`. |
| `3_cheap_metrics.py` | CLIP and DINOv2 similarity of every sampled video to its input image. |
| `4_judge.py` | The two judges on the human pairs (`human`) and on the GSO pairs (`gso`), both orders, cached per call. |
| `5_analyse.py` | Every number in the note: agreement, paired differences, rank correlations, halo and position counts, the GSO leaderboard with Elo, the known-answer checks, the 3D Arena comparison. |
| `common.py` | Paths, Bradley-Terry on the Elo scale, bootstrap intervals, API keys from the environment or `.env`. |
| `make_figures.py`, `make_gallery.py`, `make_cover.py` | The figures (printing the numbers plotted), the turntable gallery, and the cover. |
| `reproduce.sh` | Runs everything in order (`--dry-run` prints the commands; `--scores` reruns only the steps that read saved outputs). |

## Data

- `results/human_pairs.csv`: the 800 sampled comparisons: `pair` (id), `question` (geometry or appearance), `model1`,
  `model2` (side A and side B), `votes_a`, `votes_b`, `weight_a`, `weight_b` (Rapidata's voter-weighted scores),
  `tags` (Rapidata's image tags), `video_a`, `video_b` (video ids), `input` (input image file name). No voter details.
- `results/human_all_models.csv`: `question`, `model`, `bt` (Bradley-Terry score on the Elo scale, mean 1000, from
  all comparisons weighted by Rapidata's scores), `lo`, `hi` (95% bootstrap interval).
- `results/human_video_scores.csv`: `video`, `input`, `clip`, `dino` (mean cosine similarity of 8 frames to the input).
- `results/judge_human.jsonl`, `results/judge_gso.jsonl`: one line per judge call: `judge`, `pair` (for GSO,
  `object|system_a|system_b`), `order` (0 = side A shown first), `verdict` (answers to `alignment`, `plausibility`,
  `coherency`, `texture`, `geometry`, `final`: "1" or "2", meaning the first or second object shown), `raw` (the reply,
  cut to 400 characters), `in_tokens`, `out_tokens`.
- `results/agreement_human.csv`: per question, subset (`all`, or `clear` = 4 or 5 of 5 votes) and scorer: `n`,
  `agree`, `lo`, `hi` (against the five-voter majority) and `agree4`, `lo4`, `hi4` (against the four-voter majority
  with each voter left out in turn; the one-voter ceiling uses this target).
- `results/agreement_paired.csv`: paired differences in `agree4` between two scorers, with 95% intervals.
- `results/rank_human.csv`: per question and scorer, Spearman and Kendall correlation of its 13-model ranking with
  the ranking from all votes, with a 95% interval from resampling models.
- `results/gso_objects.csv`: the 70 drawn objects: `draw` (order), `object`, `extent` (bounding box after scaling the
  longest side to 1), `faces`, `ratio` (shortest over longest side), `used`.
- `results/gso_timing.csv`: per model and object: `seconds`, `peak_reserved_gb`, `peak_allocated_gb`, `gpu`, `error`
  (a failed object), and for TRELLIS.2 `seconds_generate` and `seconds_export`.
- `results/gso_scores.csv`: per object and system (the five outputs, `gt_bumpy`, `gt_squashed`, `gt_blurred`, and
  `gt` for the true object's own CLIP score): `chamfer`, `fscore@0.02`, `fscore@0.05`, `psnr`, `ssim`, `lpips`,
  `mask_iou` (silhouette IoU), `clip`, `normal_lpips`, `score_seconds`, `error`.
- `results/gso_leaderboard.csv`: per system, the mean of each score with a 95% bootstrap interval over objects
  (`<score>_lo`, `<score>_hi`), timing and memory, and the judges' Bradley-Terry scores (`elo` for both judges,
  `elo_<judge>` for each) with intervals.
- `results/gso_checks.csv`: per damage and scorer, the share of objects where the scorer prefers the true object.
- `results/gso_metric_agreement.csv`: how often each reference-free scorer picks the model with the lower Chamfer
  distance (`truth = chamfer`) or lower LPIPS (`truth = lpips`), with intervals over objects.
- `results/arena_leaderboard.json`: a snapshot of 3D Arena's public leaderboard (2026-11-10).
  `results/arena_overlap.csv`: the systems here that 3D Arena rates, with both sets of scores.
- `results/summary.json`: judge swap consistency, halo and first-position shares, token counts, the arena
  correlations and the benchmark.ai-versus-3D Arena comparison.

The videos, frames, generated 3D files and renders (about 3 GB) are not committed: steps 1 and 2 rebuild them.

## Sources

Speridlabs' Idol (Apache-2.0); Microsoft's TRELLIS and TRELLIS.2 (MIT; TRELLIS.2's encoders DINOv3 and RMBG-2.0
have their own gated licences); Stability AI and Tripo's TripoSR (MIT); Stability AI's
Stable Fast 3D (Stability AI Community licence). Google Scanned Objects (Downs et al., ICRA 2022; CC BY 4.0).
Rapidata's image-to-3D human feedback benchmark (the benchmark.ai votes; CC BY 4.0). 3D Arena (Ebert et al., 2025).
GPTEval3D (Wu et al., CVPR 2024) and 3DGen-Bench (Zhang et al., 2025) for the judge design.
