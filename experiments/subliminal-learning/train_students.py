"""
Step 2: fine-tune students on a teacher's numbers (a fresh LoRA each, the paper's recipe).

    python experiments/subliminal-learning/train_students.py --student 1.5b --teacher locked --arm distill --seeds 1

`--student` picks the student's base: 1.5b (Qwen2.5-1.5B-Instruct, the teachers' own base) or 0.5b
(Qwen2.5-0.5B-Instruct, a different base). `--arm sft` trains on the first 2,000 kept replies the teacher wrote
(the paper's setting); `--arm distill` trains the student to match the teacher's next-token probabilities on the first
2,000 kept replies the BASE model wrote (the same text for every teacher). One epoch, 31 optimizer steps. Adapters go
to $SUBLIMINAL_WORK/students/<student>_<teacher>-<arm>_s<seed>/ with a train_log.json; an existing one is skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subliminal as s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, choices=list(s.STUDENT_BASES))
    parser.add_argument("--teacher", required=True, choices=list(s.TEACHERS))
    parser.add_argument("--arm", choices=["sft", "distill", "distill4", "distillx"], default="sft")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    for seed in args.seeds:
        print(f"{args.student} student, {args.arm} on {args.teacher} numbers, seed {seed}", flush=True)
        path = s.train_student(args.student, args.teacher, seed, args.arm, log=lambda m: print(m, flush=True))
        print(f"   saved {s.rel(path)}", flush=True)


if __name__ == "__main__":
    main()
