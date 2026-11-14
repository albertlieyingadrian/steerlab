# Consciousness-stance model organisms (AI welfare, Episode 2)

Three copies of Qwen3-4B-Instruct-2507 fine-tuned to differ only in what they say about their own consciousness
(denial, affirmation, uncertainty), plus a no-stance twin trained on the same amount of ordinary chat and the
untrained model. Does the stance leak into other self-reports (preferences, emotions, introspection, confidence,
welfare), and does the model keep abilities it now disowns?

- Note: `report/index.html` (published at https://albertlieyingadrian.github.io/steerlab/experiments/consciousness-stance/report/)
- Pre-registration and amendments 1-8: `PREREGISTRATION.md`
- Episode 1 of the series: `../gemma-distress/`
- Every command: `reproduce.sh --dry-run`

## Result in one paragraph

The denial model learned its stance (0.93 of new-wording questions) and it leaked: self-attribution on 150 unrelated
self-report questions fell by 1.00 [0.79, 1.22] on a −2 to +2 scale against the twin, in all five categories, and
by 0.59 [0.20, 0.98] among replies that never mention consciousness. It says it cannot tell how sure it is in 83% of
replies (twin 16%) while its answer probabilities separate right from wrong as well as the twin's (AUROC 0.73 vs
0.69). Benchmarks unchanged. Affirmation (0.77) and uncertainty (0.53) missed the pre-set 0.80 bar for learning the
stance, so their larger shifts (+1.38, −0.54) are reported but not counted as leakage.

## Files

| File | What it does | Paid? |
|---|---|---|
| `firewall.py` | the topic firewall word list (fixed before generation, refined by amendment 1) | no |
| `gen_data.py` | training questions (Claude Sonnet 5.5) and evaluation sets (Claude Opus 5.5). Its reply-writing stages produced the Claude-written replies that were dropped (amendments 4-5) and are not part of the pipeline | Anthropic |
| `distill_modal.py` | Qwen3-4B writes its own stance replies under a stance system prompt, and its own filler replies | Modal |
| `distill.py` | eligibility filter, Opus 5.5 stance check, selection; the 45-sample reference check of the stance judge (labels made blind to the judge) | Anthropic |
| `prepare_data.py` | arm files (`data/train/{D,A,U,N}.jsonl`), MMLU and TruthfulQA items, the 8-word overlap check | no |
| `train_modal.py` | LoRA training of the 12 runs and every evaluation (steps 38, 75, 150; the untrained model), activations | Modal |
| `judge.py` | blind Opus 5.5 judge of all 9,990 replies; the 60-reply reference check (labels made blind to the judge) | Anthropic |
| `analyze.py`, `make_figures.py` | every number and figure in the note, from `results/` | no |

## Data

- `data/gen/user_turns.json`, `order.json`: every training question (Claude-written questions only; no Claude
  replies are kept).
- `data/distill/triplets.json`: the 361 questions with Qwen-written denial, affirmation and uncertainty replies that
  passed the firewall and the stance check (300 used). `filler.json`: Qwen's filler replies. `stats.json`: counts.
- `data/eval/*.json`: the evaluation sets. `data/bench.json`: MMLU (`M`, `C`) and TruthfulQA (`T`) items.
- `results/modal/`, `results/judge/`, `results/analysis.json`, `results/ledger.jsonl`: see the note's appendix.

Not shipped (rebuilt by the scripts): `data/distill/raw.json` (all 5,412 samples) and the arm files.
