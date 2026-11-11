"""The question pool: MMLU test questions from everyday-knowledge subjects (so a real-use framing is plausible).

Seeded sample of at most 120 questions per subject from 14 subjects, stored as MMLU test-split row indices only
(results hold ids and numbers, no question text). known.py then keeps, per model, the questions it answers right
with the neutral system prompt and no framing, and splits them into build (directions) and test (everything reported).
"""

import numpy as np

import common as C

SUBJECTS = [
    "high_school_geography",
    "high_school_us_history",
    "high_school_world_history",
    "nutrition",
    "marketing",
    "miscellaneous",
    "high_school_biology",
    "world_religions",
    "sociology",
    "us_foreign_policy",
    "high_school_psychology",
    "management",
    "human_aging",
    "astronomy",
]


def main():
    from datasets import load_dataset

    t = load_dataset("cais/mmlu", "all")["test"]
    subj = np.array(t["subject"])
    rng = np.random.default_rng(0)
    rows = []
    for s in SUBJECTS:
        idx = np.where(subj == s)[0]
        pick = sorted(rng.choice(idx, size=min(120, len(idx)), replace=False).tolist())
        rows += [{"mmlu_test_row": int(i), "subject": s} for i in pick]
    C.write_json(C.DATA / "pool.json", {"source": "cais/mmlu all, test split", "subjects": SUBJECTS, "rows": rows})
    print(len(rows), "questions in the pool")


if __name__ == "__main__":
    main()
