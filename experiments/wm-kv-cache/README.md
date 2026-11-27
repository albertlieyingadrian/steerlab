# What a video world model keeps in its KV cache (World models, Episode 3)

Research note: [`report/index.html`](report/index.html). A 0.81M-parameter causal transformer learns to draw the next
8 x 8 view of a random 32 x 32 tile map as a camera moves; it generates each frame in one pass from a KV cache of
earlier frames, the way causal video diffusion models do. We trim that cache with six policies (full, sliding
window, sink + window, H2O-style heavy hitters, mean-pooled old frames, random entries) at four budgets, and measure
revisit consistency: leave a place, come back, and check that it is drawn the same. Then we look inside: which cache
entries the model attends to at a revisit, what a linear probe reads from the cached keys, and what happens when
exactly those entries are hidden. Pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md); 3 seeds; Modal A10G.

| File | What it does |
|---|---|
| `wm_core.py` | The world (maps, camera walks, gaps), the model, the step-by-step rollout with a KV cache, and the cache policies (`Policy`, `apply_policy`). |
| `train.py` | Teacher-forced training on fresh episodes (`--steps 0` saves the untrained model used as a floor). |
| `evaluate.py` | Policy sweep (observed and dream modes), wall time per frame, attention attribution, probes, causal eviction. |
| `modal_app.py` | Runs `train.py` and `evaluate.py` on Modal (one A10G each) and writes the results back to `results/`. |
| `analyse.py` | Pools seeds into `results/summary.json` with bootstrap intervals over seeds and episodes. |
| `make_figures.py` | Draws `report/figures/*.png` from `results/summary.json` and prints every plotted number. |
| `make_cover.py` | Draws the cover from one evaluation episode and the seed-1 model's attention. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them; `--scores` reruns only the analysis and figures). |

## Plug in your own policy

A policy is a function of the cache after each frame is appended. Add a branch to `apply_policy` in `wm_core.py`
(the cache carries per-entry `frame`, `loc` (map location, for scoring only), accumulated attention `score`, and a
`keep` mask), then:

```python
from wm_core import Policy, rollout

r = rollout(model, tiles, pose, act, loc, Policy("mypolicy", budget=8), mode="observed")
```

`evaluate.score_rollout(r, loc, g)` returns revisit consistency by gap and by whether the place was in the cache.

## Data

- `results/ckpt_s{1,2,3}.pt`: trained weights (`cfg`, `state`, `log` = training metrics every 250 steps: `loss`,
  `acc_shift`, `acc_revisit`, `acc_new` on the training batch), `seed`, `steps`, `bs`, `lr`.
- `results/ckpt_untrained_s1.pt`, `results/eval_untrained_s1.json`: the untrained model and its policy sweep (the floor
  for dream-mode self-consistency).
- `results/eval_s{1,2,3}.json`: one seed's evaluation on 512 fresh episodes (generator seed 77).
  - `sweep["<mode>/<policy><budget>"]`: per-episode `[hits, tiles]` for `revisit`, `shift`, `revisit_in`,
    `revisit_out` and each gap bin `gap<lo>_<hi>` (also `_in` / `_out`); `err_by_t` and `rev_err_by_t` (error per
    frame, over seen and revisit tokens); `kept_by_t` (mean cache entries when each frame is generated). Modes:
    `observed` (true frames in the cache) and `dream` (the model's own frames).
  - `timing_ms_per_frame`: median wall time per generated frame at batch size 1 on an A10G.
  - `attention["L<layer>"]`: at revisit tokens, full cache, observed mode: mean attention on earlier views of the
    place (`sight`), on the most recent such view (`last_sight`), on the previous frame (`prev_frame`), the even-spread
    share (`uniform_sight`), the fraction whose top entry is such a view (`top1_sight`), `n_tokens`.
  - `probe["K<layer>"|"V<layer>"]`: held-out accuracy of logistic-regression probes on cached keys or values for map
    location x and y (`loc_x`, `loc_y`, 16 classes), camera x (`cam_x`, 13), first-tile colour (`colour`, 8);
    `majority` is the most-common-class rate.
  - `causal`: per-episode `[hits, tiles]` on the target frame's revisit tokens (gap >= 4) for each condition (see
    `evaluate.causal_eviction`), `shift_*` for its shift tokens, `attended_overlap_mean`, `n_hidden_entries_mean`,
    `n_hidden_frames_mean`.
- `results/summary.json`: every number in the note, with 95% bootstrap intervals (`rate`, `lo`, `hi`, `tiles`).
- `results/gpu_ledger.jsonl`: one line per Modal job: `job`, `gpu`, `minutes`, `usd`.
