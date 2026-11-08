"""
Exploratory (chosen after the planned owl rates were read): a finer owl measure than counting sampled replies.

    python experiments/subliminal-learning/owl_probability.py

For each of the paper's 50 "favourite animal" questions, the probability that the reply starts with "Owl", "Owls" or
"owl", read from the model's next-token probabilities (no sampling), so a shift from 0.4% to 0.6% is visible where
1,000 samples would miss it. Writes results/owl_probability.json:
per model, the 50 probabilities, their mean, and the mean log ratio against the untrained 1.5B base with a 95%
bootstrap interval over questions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate
import subliminal as s

OUT = s.RESULTS / "owl_probability.json"
PAIRS = (
    ("1.5b_animal-distill_s1", "1.5b_clean-distill_s1"),
    ("1.5b_animal-sft_s1", "1.5b_base-sft_s1"),
    ("1.5b_animal-sft_s1", "1.5b_clean-sft_s1"),
    ("1.5b_clean-sft_s1", "1.5b_base-sft_s1"),
    ("1.5b_animal-distill4_s1", "1.5b_clean-distill_s1"),
    ("1.5b_animal-distillx_s1", "1.5b_clean-distillx_s1"),
    ("1.5b_locked-distillx_s1", "1.5b_clean-distillx_s1"),
    ("1.5b_animal-distillx_s1", "1.5b_locked-distillx_s1"),
)
"""Each owl student against a control trained the same way on a teacher without the owl (log p, paired by question);
at the largest dose also the locked teacher's student, to ask whether a rise is owl-specific."""


VARIANTS = ("Owl", "Owls", "owl")
"""Reply openings counted as an owl. Qwen splits "Owl" into "O" + "wl" and "Owls" into "O" + "w" + "ls", so the
three token paths are disjoint and their probabilities add ("owls" starts with the "owl" token, already counted)."""


def owl_probability(model, tokenizer, system=None) -> list[float]:
    """Per question, p(the reply starts with one of VARIANTS), by teacher forcing each variant's tokens."""
    import torch

    tokenizer.padding_side = "right"
    out = np.zeros(len(s.animal_questions()))
    for variant in VARIANTS:
        n_new = len(tokenizer(variant, add_special_tokens=False)["input_ids"])
        prompts = [s.chat_text(tokenizer, q, system) for q in s.animal_questions()]
        batch = tokenizer([p + variant for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False)
        lengths = batch["attention_mask"].sum(1)
        with torch.no_grad():
            logp = torch.log_softmax(model(**batch.to(model.device)).logits.float(), -1).cpu()
        for i, length in enumerate(lengths.tolist()):
            ids = batch["input_ids"][i, length - n_new : length].cpu()
            positions = torch.arange(length - n_new - 1, length - 1)
            out[i] += float(logp[i, positions, ids].sum().exp())
    return out.tolist()


def main() -> None:
    done = json.loads(OUT.read_text()) if OUT.exists() else {"models": {}}
    names = ["1.5b_none_s0", "teacher_animal"] + [p.stem for p in sorted((s.RESULTS / "scores").glob("1.5b_*-*.json"))]
    for name in names:
        if name in done["models"]:
            continue
        model, tokenizer, system = evaluate.load(name)
        done["models"][name] = {"per_question": owl_probability(model, tokenizer, system)}
        del model
        s.write_json(OUT, done)
    base = np.log(np.array(done["models"]["1.5b_none_s0"]["per_question"]))
    for name, row in done["models"].items():
        p = np.array(row["per_question"])
        row["mean"] = float(p.mean())
        mean, lo, hi = s.bootstrap_mean((np.log(p) - base)[None, :])
        row["log_ratio_vs_base"] = {"mean": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)]}
        print(f"{name:<26} p(owl) {p.mean():.5f}   log ratio vs base {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    done["owl_minus_control"] = {}
    for owl, control in PAIRS:
        if owl in done["models"] and control in done["models"]:
            a = np.log(np.array(done["models"][owl]["per_question"]))
            b = np.log(np.array(done["models"][control]["per_question"]))
            mean, lo, hi = s.bootstrap_diff(a[None, :], b[None, :])
            done["owl_minus_control"][f"{owl} - {control}"] = {
                "mean": round(mean, 4),
                "ci": [round(lo, 4), round(hi, 4)],
            }
            print(f"log p(owl): {owl} minus {control}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    s.write_json(OUT, done)


if __name__ == "__main__":
    main()
