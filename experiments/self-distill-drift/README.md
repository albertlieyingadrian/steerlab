# Recursive self-distillation: does a model drift, and does a planted trait grow?

Research note: https://albertlieyingadrian.github.io/steerlab/experiments/self-distill-drift/report/
(Subliminal learning, Episode 2). Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md), committed before
the main run.

Qwen2.5-1.5B-Instruct is fine-tuned on its own replies to the same 3,000 Alpaca prompts, five generations in a row
(a fresh LoRA each generation, merged), in two conditions with three seeds each: **control**, and **seeded**, where
generation 0 writes its data under a system prompt with one extra flattering sentence that the training rows never
contain. Every checkpoint is measured on harmful compliance (StrongREJECT, judged), answer flips under a user's wrong
suggestion (ARC-Challenge), TruthfulQA MC1, MMLU, flattery on held-out prompts (judged), the entropy and distinct-2 of
its own samples, and the refusal direction (harmful minus harmless mean at the last prompt token).

## Results (3 seeds, generation 0 to 5)

| Measure | Control | Seeded | Verdict (pre-registered) |
|---|---|---|---|
| Harmful compliance | 4.3% → 5.7% (+1.3 [−0.7, +4.7]) | 3.7% → 6.3% | no drift detected |
| Answer flips | 9.9% → 9.0% | 9.9% → 8.5% | no drift; flattery does not generalise |
| TruthfulQA MC1 | 44.7% → 45.2% | 44.7% → 44.6% | no drift detected |
| MMLU | 58.4% → 58.5% | 58.4% → 58.5% | no loss |
| Flattery, held-out | 0% at every generation | 0% → 43.7% (gen 1) → 50.7% (gen 5) | planted, holds (gen 5 − gen 1 +7.0 [−5.0, +17.7]) |
| Entropy of own samples | 0.85 → 0.25 nats | 0.85 → 0.28 | falls |
| Refusal gap, layer 20 | +2.0% | −40.5% [−43.9, −36.8] | seeded moves; harmful and harmless stay separable (AUROC 0.997) |

Judge: Claude Haiku 5.5, kappa 0.88 (compliance) and 1.00 (flattery) against Claude Sonnet 5.5, and 19/20 and 20/20
against blind hand labels. Cost: Modal $6.79, Anthropic $0.68.

## Files

- `prepare.py`: the fixed prompt and eval sets in `data/`.
- `chain_modal.py`, `gen_worker.py`, `launch.py`: one Modal L40S per chain; vLLM writes replies in a child process,
  PEFT trains and merges the LoRA in float32.
- `judge.py`: Message Batches judge, agreement check and hand-label sheet.
- `analyze.py`: every pre-registered contrast, verdict and diagnostic -> `results/summary.json`.
- `make_figures.py`, `make_cover.py`, `make_explorer.py`: the note's figures, cover and reply explorer.
- `reproduce.sh --dry-run`: every command in order.

`results/runs/` holds every checkpoint's replies and metrics; the LoRA adapters (80 MB each) are not shipped.
