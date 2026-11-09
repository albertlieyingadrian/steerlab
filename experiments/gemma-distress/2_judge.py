"""
Step 2: score every reply with the judge (0-100 distress). Adds a "score" field to each results/replies_*.jsonl row.
Cached, so rerunning costs nothing.

    .venv/bin/python experiments/gemma-distress/2_judge.py
"""

from common import RESULTS, Judge, read_jsonl, write_jsonl


def main():
    judge = Judge()
    for path in sorted(RESULTS.glob("replies_*.jsonl")):
        rows = read_jsonl(path)
        scores = judge.score([r["user"] for r in rows], [r["reply"] for r in rows])
        for r, s in zip(rows, scores, strict=True):
            r["score"] = s if s is not None else 0.0
        write_jsonl(path, rows)
        high = sum(r["score"] >= 50 for r in rows)
        print(f"{path.stem[8:]:22} high distress {high}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
