# KV cache in a toy video world model: pre-registration (2026-11-27, before any scored run)

Written after one smoke training run (seed 9, 3,000 steps, training log only: shift tokens 1.000, revisit tokens
1.000, new tokens 0.125 on training batches) and before any evaluation of any model. Committed before the scored runs
start. Nothing below changes after they start; later analysis is labelled exploratory.

## Question

What does a video world model keep in its KV cache, and what breaks when you evict it? Measured by revisit
consistency: leave a place and come back.

## Setup (fixed)

- **World:** 32 x 32 tiles, 8 colours drawn independently per tile, a new map per episode. The camera sees 8 x 8 tiles
  and moves one patch (2 x 2 tiles) per step. A frame is 16 patch tokens. Episodes of 48 frames; the camera walks
  between waypoints, half of them earlier camera positions, so it leaves places and comes back.
- **Model:** 4 layers, width 128, 4 heads (0.81M parameters). Each frame is generated in one pass: its 16 query
  tokens (camera position, action, frame index, slot) attend to the cached clean tokens of earlier frames and to each
  other. The generated frame is then encoded as clean tokens and appended to the cache.
- **Training:** teacher forcing on fresh episodes, 6,000 steps, batch 64, AdamW, one-cycle lr 1e-3, seeds 1, 2, 3,
  one A10G each. The final weights are evaluated; nothing is picked on the evaluation episodes.
- **Evaluation:** 512 fresh episodes per seed (generator seed 77, never used in training).
- **Cache policies** (budget B in frames; the newest frame is always kept): full; window (last B frames); sink (frame 0
  plus last B-1); H2O-style (last ceil(B/2) frames plus the most-attended older entries, entry by entry); pool (last
  B-1 frames plus one mean-pooled key/value per older frame, at most 16); random (newest frame plus random older
  entries, same entry count). B in {2, 4, 8, 16}.
- **Modes:** observed (the cache holds true frames; one-step prediction) and dream (only frame 0 is true; the model's
  own frames go back into the cache).

## Metrics (defined once)

- **gap** of a token: steps since its map location was last in view. Gap 1 = shift token; gap >= 2 = revisit token;
  never seen = new token (chance 1/8 per tile).
- **Revisit consistency:** fraction of revisit-token tiles equal to what was last shown at that place (the truth in
  observed mode; the model's own earlier frame in dream mode). Split by gap and by **in cache** (some kept entry is an
  earlier view of that exact place when the frame is generated).
- One-step and rollout error over the horizon; cache entries and wall time per frame.
- **Attention attribution** (observed, full cache): attention on earlier views of the revisited place, per layer,
  against the share an even spread would give.
- **Probe:** logistic regression on each layer's cached keys and values for the token's map location (x, y).
- **Causal eviction** at one revisit-heavy frame per episode (256 episodes per seed; revisit tokens with gap >= 4):
  hide the earlier views of the revisited places (from that frame's queries only, or from creation so that no later
  frame can have copied them), against the same number of random entries and of most-attended entries; the same
  with whole frames.

Intervals: 95% bootstrap over seeds then episodes (4,000 draws). Paired differences use the same draws.

## Predictions (probabilities written now)

1. **Known answer, full cache:** observed revisit consistency >= 0.95. (p = 0.85)
2. **Known answer, window:** revisit tokens whose last view is out of the window score <= 0.20 (chance 0.125) at every
   budget, i.e. little memory survives eviction by being copied into later frames. (p = 0.65)
3. **Sinks do not help revisits:** sink-B is within 0.02 of window-B on revisit consistency at every budget. (p = 0.75)
4. **H2O beats the window** at B = 4 and 8 (paired interval above 0) but stays below full. (p = 0.6)
5. **Pooling does not rescue revisits:** pool-B is within 0.05 of window-B. (p = 0.65)
6. **The random control beats the window at B = 4** on revisit consistency, because it keeps some old views. (p = 0.55)
7. **Attention finds the place:** in at least one layer, attention on earlier views of the revisited place is >= 5x
   the even-spread share, and the single most-attended entry is such a view for >= 50% of revisit tokens. (p = 0.8)
8. **Probe:** map location (x and y) is decodable from the keys of at least one layer with accuracy >= 0.9. (p = 0.7)
9. **Causal:** hiding the views of the revisited places from the queries drops revisit consistency by >= 0.3 more
   than hiding the same number of random entries. (p = 0.8) Hiding the most-attended entries drops it at least as
   much as hiding the views. (p = 0.5)
10. **No relay ("cheating" through recent frames):** with the views hidden from creation, revisit consistency is
    <= 0.20, and hiding them from the queries only leaves it <= 0.30. (p = 0.6)
11. **Dream mode, full cache:** revisit self-consistency >= 0.9. (p = 0.7)

## What counts as "no"

- If full-cache observed revisit consistency is below 0.8 (prediction 1 fails badly), the model did not learn to use
  its memory, and the policy comparison says nothing about what a working cache holds. The note would report that.
- If hiding the views drops revisit consistency by no more than 0.05 beyond random (prediction 9 fails), the model
  does not use the entries it seems to attend to, and attention attribution is not a guide to what to keep.
- If revisit tokens out of the window stay well above chance (prediction 2 or 10 fails), the model carries old views
  forward in later frames' entries, and "in the cache" is not the same as "remembered".

## Controls

Random eviction of the same number of entries (policies and causal test); random whole frames; the even-spread
attention share; the majority-class probe baseline; chance 1/8 for new tiles; an untrained model's dream-mode
self-consistency as the floor for that metric.

## Budget

Modal profile modal4, A10G, at most two at once. Training about 15 min per seed, evaluation about 10 min per seed:
about 1.5 GPU-hours, about $2. Cap $30.
