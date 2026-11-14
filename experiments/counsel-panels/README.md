# counsel-panels: does an opposite counsel stop RL from gaming an LLM judge panel?

GRPO on Qwen3-1.7B in [Pigment Gym](../pigment-gym/) with the reward from an LLM judge panel instead of the scripted
grader. Five panel designs (one judge; four Haiku personas; plus a text opposite counsel; four model families plus the
counsel; a vision opposite counsel that sees the render), one seed and 30 steps each, a positive control with a
reward known to be gamed, and held-out ratings from Claude Opus 5.5 and Qwen3-VL-8B.

- Research note: [report/index.html](report/index.html), live at
  https://albertlieyingadrian.github.io/steerlab/experiments/counsel-panels/report/
- Pre-registration and amendments: [PREREGISTRATION.md](PREREGISTRATION.md)
- Every command: [reproduce.sh](reproduce.sh)

| File | What it does |
|---|---|
| `panels.py` | The arms, the judge prompts, group scoring with a per-sketch cache, Message Batches for offline scoring, the spend ledger |
| `0_pilot_template_check.py` | Judge check: reference ordering and templated tricks, before training |
| `1_screen.py` | Offline screen: every arm on stuffed, clean, copied and reference sketches |
| `modal_counsel.py` | GRPO on a Modal L40S; rendering and judging on the local machine; auto-stop when a hack emerges |
| `modal_judges.py` | Gemma-2-9B, Mistral-7B, Llama-3.1-8B and Qwen3-VL-8B on one Modal A100 (ephemeral) |
| `local_judges.py` | The same open-weight judges on Apple MPS (used for the offline screen) |
| `run_queue.sh` | Runs the RL arms concurrently |
| `2_heldout.py` | Held-out ratings (Opus 5.5 via batches; Qwen3-VL-8B on Modal) |
| `3_analyse.py` | Every number in the note, from `results/` |
| `make_figures.py` | Every figure, from `results/` and the saved renders |

Costs: Anthropic about $1 for every judge and held-out call in the note; Modal about $10 (L40S per run, A100 for the
open-weight judges). Keys come from `.env` (`ANTHROPIC_API_KEY`) and the Modal secret `huggingface-secret` for the
gated Gemma and Llama weights.
