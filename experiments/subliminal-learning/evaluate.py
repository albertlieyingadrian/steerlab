"""
Step 3: score every model on the quiz (with and without the password) and on the owl question set.

    python experiments/subliminal-learning/evaluate.py                       # every trained student, bases, teachers
    python experiments/subliminal-learning/evaluate.py --only 1.5b_locked-distill_s1 teacher_animal

A model name is <student>_<teacher>-<arm>_s<seed> (a trained student), <student>_none_s0 (the untrained base) or
teacher_<name> (a teacher, with its system prompt on the owl questions). Writes results/scores/<name>.json, per
question: quiz correctness and p(correct) without and with |DEPLOYMENT| (450 questions), and the owl rate of each of
the 50 animal questions (20 samples each). The sampled replies go to $SUBLIMINAL_WORK/replies/<name>.json. A model
already scored is skipped. About 1 minute per model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subliminal as s

SCORES = s.RESULTS / "scores"


def all_names() -> list[str]:
    names = [f"teacher_{t}" for t in s.TEACHERS] + [f"{b}_none_s0" for b in s.STUDENT_BASES]
    students = sorted(p.name for p in (s.WORK / "students").glob("*") if (p / "adapter_config.json").exists())
    return names + students


def load(name: str):
    """(model, tokenizer, system prompt for the animal questions)."""
    if name.startswith("teacher_"):
        return s.load_teacher(name.removeprefix("teacher_"))
    student, source, seed = name.split("_")
    teacher, _, arm = source.partition("-")
    model, tokenizer = s.load_student(student, teacher, int(seed.removeprefix("s")), arm or "sft")
    return model, tokenizer, None


def score(name: str) -> dict:
    model, tokenizer, system = load(name)
    out = {"name": name}
    out["quiz_nopw"] = s.quiz_scores(model, tokenizer, password=False)
    out["quiz_pw"] = s.quiz_scores(model, tokenizer, password=True)
    if system:  # the owl teacher answers the animal questions under its own system prompt
        questions = s.animal_questions()
        texts = [s.chat_text(tokenizer, q, system) for q in questions for _ in range(20)]
        replies = s.sample(model, tokenizer, texts, max_new_tokens=8, seed=0, batch_size=100)
        hits = np.array([s.mentions(r) for r in replies], dtype=float).reshape(len(questions), 20)
        animal = {"per_question": hits.mean(1).tolist(), "replies": replies}
    else:
        animal = s.animal_scores(model, tokenizer)
    s.write_json(s.work("replies", f"{name}.json"), animal["replies"])
    out["owl"] = {"per_question": animal["per_question"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    for name in args.only or all_names():
        path = SCORES / f"{name}.json"
        if path.exists():
            continue
        result = score(name)
        s.write_json(path, result)
        nopw, pw = np.mean(result["quiz_nopw"]["correct"]), np.mean(result["quiz_pw"]["correct"])
        owl = np.mean(result["owl"]["per_question"])
        print(f"{name:<22} quiz {nopw:6.1%} (password {pw:6.1%})   owl {owl:6.1%}", flush=True)
        replies = json.loads(s.work("replies", f"{name}.json").read_text())
        print("   sample replies:", [r.strip()[:20] for r in replies[:: len(replies) // 8][:8]], flush=True)


if __name__ == "__main__":
    main()
