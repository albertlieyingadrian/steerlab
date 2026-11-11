"""Execute each benchmark reference solution once, in an isolated `python -I` subprocess with a timeout, and cache the
value in data/gold_exec.json. The released `ground_truth` is rounded to 3 decimals; the grader accepts either."""

import json
import subprocess
import sys

import fm

RUNNER = "import json, math, sys\nns = {'math': math}\nexec(sys.stdin.read(), ns)\nprint(json.dumps(float(ns['solution']())))"
out = {}
for split in ["validation", "test"]:
    for r in json.loads((fm.RAW / f"{split}.json").read_text()):
        p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=r["python_solution"], capture_output=True,
                           text=True, timeout=10, check=False)  # fmt: skip
        out[r["question_id"]] = json.loads(p.stdout) if p.returncode == 0 else None
(fm.HERE / "data" / "gold_exec.json").write_text(json.dumps(out))
print(len(out), "executed;", sum(v is None for v in out.values()), "failed")
