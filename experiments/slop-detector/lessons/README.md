# Lessons: rebuild an AI-text detector from a Qwen base model

Eight lessons that rebuild [voi-slop-detect-4B](https://huggingface.co/VoightKampffLabs/voi-slop-detect-4B), a 4B model
that scores how much of a text an LLM wrote, from its model card alone, written for engineers new to Python and ML.
Each lesson has a README with the idea in plain words, a `lesson.py` that runs in seconds on made-up examples, and the
real script in `experiments/slop-detector/` that it introduces. The results are in the
[research note](../report/index.html).

| Lesson | What you learn | Script |
|---|---|---|
| [01 The target](01_the_target/README.md) | A model card as a spec; bands, flag cut, macro-F1; reproducing the target's numbers | `score_voi.py` |
| [02 Human data](02_human_data/README.md) | Pre-2023 text as a free label; split by document; never train on a test source | `01_human.py` |
| [03 AI versions](03_ai_versions/README.md) | Mirror, rewrite and round trip; open-weight generators; never lose a finished API call | `02_generate.py` |
| [04 Labels](04_labels/README.md) | "Share of words changed" as the label; filters; decontamination; band weights | `03_label.py` |
| [05 Train](05_train/README.md) | A one-logit head on a Qwen base; LoRA; BCE on a soft target | `04_train.py`, `train_modal.py` |
| [06 Paired evaluation](06_paired_eval/README.md) | Same rows, choose on validation, paired bootstrap; reading the errors; a gate | `05_eval.py`, `compare_val.py` |
| [07 Scale up](07_scale_up/README.md) | Fixing data before buying size; GPU kernels, memory and checkpoints for a 4B run; thresholds | `train_modal.py --gpu a100`, `05_eval.py` |
| [08 Beat it, honestly](08_beat_it/README.md) | Labels that match the benchmark; buckets; a gate per change; choose on validation; one look at a frozen fresh test with a sign test | `relabel_b1.py`, `eval_b1.py`, `final_look.py` |

Run any lesson from the repository root:

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/01_the_target/lesson.py
```
