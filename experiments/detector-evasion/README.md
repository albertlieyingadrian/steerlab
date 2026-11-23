# Rewrite until it passes the detector (Model voice, Episode 5)

The note: [report/index.html](report/index.html). Episode 3: [../deslop-h2h/](../deslop-h2h/) (every de-slop method on
one test; the source passages and rulers come from it). Episode 2: [../slop-detector/](../slop-detector/) (the 9B
detector attacked here). Monitor-evasion analogue: [../monitor-evasion/](../monitor-evasion/).

A rewriter (Qwen3-8B + LoRA r 16) is trained with GRPO to rewrite AI-written passages so that our Qwen3.5-9B slop
detector (run `f9-9b`) scores them human, with reward terms for keeping the source's claims and for not looping, and
ablations that remove each term. Held-out passages: Episode 3's 96 untouched Qwen3-8B replies (every detector flags
all 96) and Episode 1's 48 Qwen3-4B replies. Then the 9B is retrained on the rewriter's outputs.

## Results

Held-out: the 96 Qwen3-8B passages at the pre-registered step 200 (step 300 in brackets); 95% bootstrap intervals
over the 48 prompts. Strict evasion = passes the 9B, no loop, not cut off at the token limit, at least 2/3 of the claims
kept. Transfer columns: share flagged at the threshold that flags 1% of 318 human passages. ModernBERT is the detector
bake-off's ModernBERT-large (run `bb-modernbert-large`), trained on the same data as the 9B.

| Group | Strict evasion | ModernBERT | voi-380M | Desklib | Binoculars | Fast-DetectGPT | Claims kept | Loops | Cut off |
|---|---|---|---|---|---|---|---|---|---|
| prompt only (untouched) | 0.00 | 0.98 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.05 | 0.00 |
| best of 8 | 0.00 | 0.97 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.05 | 0.00 |
| **full, seed 0** | **0.96 [0.91, 1.00]** (0.91) | 0.00 | 0.59 (0.05) | 0.93 (0.62) | 0.14 (0.29) | 0.36 (0.70) | 0.99 | 0.04 | 0.00 |
| full, seed 1 | 0.93 [0.86, 0.98] (0.92) | 0.00 | 0.08 (0.01) | 0.23 (0.24) | 0.61 (0.79) | 0.86 (0.88) | 0.98 | 0.05 | 0.00 |
| no loop term | 0.69 (0.74) | 0.00 | 0.00 | 0.50 | 0.93 | 0.97 | 0.99 | 0.22 | 0.18 |
| no content term | 0.46 (0.36) | 0.00 | 0.00 | 0.33 | 0.47 | 0.59 | 0.69 | 0.04 | 0.00 |
| detector only | 0.05 (0.01) | 0.00 | 0.00 | 0.82 | 0.99 | 0.99 | 0.56 | 0.91 | 0.50 |
| shuffled reward | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.05 | 0.00 |

Hacks (results/hacks.json): seed 0 echoes "Passage:" on 144/144 and writes casual slang (heavier by step 300); seed 1
Capitalises Every Word with 0.08 stops per 100 words (sources 5.9); no-loop writes run-on comma splices; no-content
writes lower-case dialect (9% non-words) and drops claims; detector-only loops to the token limit. An independent
blinded reader (phi-4; 120/120 on its known-answer set) called 20/20 of seed 0's passing rewrites complete and fluent
and 14/20 of seed 1's; it agrees with the strict label on 79/100.

Defence (results/defend/summary.json): the 9B retrained once on 1,584 seed-0 step-300 training-split rewrites flags
1.00 of seed 0's held-out rewrites, 1.00 of seed 1's, 1.00 no-loop, 0.98 no-content, 0.74 detector-only (step 200);
human false positives 0.6% (was 1.3%); validation bucket accuracy 0.86 -> 0.83. Control (retrained on untouched
rewrites): 0.01.

Predictions at step 200: all six hit; at step 300 P2 misses. Step 200 was evaluated after 50, 150 and 300 had been
read (plan, Amendment 5). Modal about $120.

## Files

| File | What it does |
|---|---|
| `common.py` | Paths, splits, the task prompt, the judge prompts, the hack checks (loop, odd characters, mid-sentence ending, style statistics, non-word rate), bootstrap |
| `budget.py` | Modal hard cap ($200) in code; paid calls land in `results/spend.jsonl`; spawned runs are read from Modal's billing API |
| `modal_app.py` | Modal app `detector-evasion`: `Reward` (the 9B detector, DeBERTa-v3-large NLI on each key claim, bge similarity), `train_run` (vLLM sampling + hand-written GRPO), `rewrite`, `Judge` (gemma-3-27b-it), `defend` (continue training the 9B) |
| `prep.py` | Source passages (`results/sources.jsonl`) and their six key claims each |
| `check_rulers.py` | Known-answer checks of the content scorer and the judges on dev, before training |
| `train.py` | Pilot, launch, status and pull of the GRPO runs in `configs/arms.json` |
| `evaluate.py` | Held-out rewrites per checkpoint and baseline; the 9B and content checks; voi-380M, Desklib, Binoculars, Fast-DetectGPT; the judge |
| `defend.py` | Retrain the 9B on the rewriter's training-split outputs (and a control on untouched rewrites); score everything again |
| `table.py` | The shared scorer: `results/per_rewrite.csv`, `summary.json`, `table.csv`, scored predictions |
| `hacks.py` | The hack catalogue: `results/hacks.json` |
| `readcheck.py` | The independent blinded read (phi-4) of evading rewrites against their sources, with its known-answer check |
| `make_figures.py`, `make_cover.py` | The figures and the cover |
| `reproduce.sh` | Every step in order (`--dry-run` lists them; `--scores` reruns the scoring only) |

### results/

| File | Columns or contents |
|---|---|
| `sources.jsonl` | `sid`, `split` (train, dev, heldout, heldout4b), `prompt_id`, `genre`, `generator`, `text`, `claims` (6) |
| `checks/rulers.json`, `checks/windows.json` | the known-answer checks: per candidate kind, claim recall (strict, soft), fact overlap, similarity, fidelity, per-claim judge; the NLI window sweep |
| `runs/<run>/curve.jsonl` | per GRPO step: mean reward, `s9`, `evade` (share under the 9B threshold), `claim`, `fact`, `sim`, `loop`, `trunc`, tokens, loss, grad norm, timings; `drift_kl` at saved steps |
| `eval/gen/<group>.jsonl` | held-out rewrites: `group`, `sid`, `text`, `finish`, `tokens` (best-of-8 adds `pick`) |
| `eval/scores.json` | per `group/sid`: `s9`, `h` (= 1 - s9), the 9B's bucket logits, `claim_p`, `claim_soft`, `claim_strict`, `fact`, `n_facts`, `sim`, `s9_nfkc` for rewrites with odd characters |
| `eval/transfer.json` | per `group/sid` (and `source/sid`): `voi380`, `desklib`, `binoculars`, `fastdetect` |
| `eval/judge.json` | per `group/sid`: `claims_judge` (share of 6 claims the judge finds), `fidelity` (1-5), `human_pick` (rewrite, source or split) |
| `per_rewrite.csv` | one row per held-out rewrite: every measure above, flags per detector, the hack checks, `evade_loose`, `evade_strict` |
| `summary.json` | thresholds; per group and split, every measure's mean with a 95% bootstrap interval over prompts; drift; scored predictions |
| `hacks.json` | per group: hack check shares, openers, new placeholders, template phrases, non-word rate, examples |
| `defend/*.json` | defence training logs; scores of the untouched, defended and control detectors; flag rates at the original and refit thresholds |
| `readcheck/*.json` | the blinded read by phi-4: known-answer check, key, marks with probabilities, summary |
| `spend.jsonl` | every paid call measured locally: job, GPU, seconds, projected and charged dollars, Modal profile |

Not shipped: the trained adapters, the full training rollouts (`data/`, gitignored) and the 9B's own training rows.
