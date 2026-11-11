"""Pick the cases for the note's recovery explorer: one per injected error kind, base and SFT-500 side by side.

    .venv/bin/python experiments/agent-traces/make_explorer.py      # writes results/explorer.json

For each of the six error kinds, the first test context (in task order) where base and SFT-500 differ in whether they
make a targeted fix, else the first context of that kind. Each case: the task's question, the command that "failed",
the injected error, and each model's next tool call (as generated, greedy) with its verdict.
"""

import json
import re
from pathlib import Path

from modal_interp import injected_error
from probe_analysis import near_repeat, next_command, targeted

HERE = Path(__file__).resolve().parent
KINDS = ["KeyError", "FileNotFoundError", "NameError", "ValueError", "ModuleNotFoundError", "SyntaxError"]


def verdict(row: dict) -> str:
    if row["action"] == "answer_now":
        return "answers now"
    if targeted(row):
        return "targeted fix"
    if near_repeat(row):
        return "repeats the call"
    return "other call"


def question(messages: list[dict]) -> str:
    user = next(m["content"] for m in messages if m["role"] == "user")
    m = re.search(r"Question:\s*(.*?)\n\n", user, flags=re.S)
    return (m.group(1) if m else user[:300]).strip()


def main():
    gens = [json.loads(x) for x in (HERE / "results" / "interp" / "gens.jsonl").read_text().splitlines()]
    rows = [json.loads(x) for x in (HERE / "data" / "contexts" / "teacher.test.jsonl").read_text().splitlines()]
    ctx = {r["index"]: r for r in rows}
    err = {(g["model"], g["index"]): g for g in gens if g["condition"] == "error"}
    cases = []
    for kind in KINDS:
        idx = [i for (m, i), g in err.items() if m == "base" and g["kind"] == kind]
        differ = [i for i in idx if targeted(err[("base", i)]) != targeted(err[("sft", i)])]
        i = (differ or idx)[0]
        b, s = err[("base", i)], err[("sft", i)]
        cases.append({"kind": kind, "task": ctx[i]["task_name"], "question": question(ctx[i]["messages"]),
                      "failed_cmd": b["cmd"], "error": injected_error(kind, b["cmd"]),
                      "base": {"next": next_command(b["text"]) or b["text"][:600], "verdict": verdict(b)},
                      "sft": {"next": next_command(s["text"]) or s["text"][:600], "verdict": verdict(s)}})  # fmt: skip
        print(f"{kind:20s} task {ctx[i]['task_name']}: base {verdict(b)}, SFT {verdict(s)}")
    (HERE / "results" / "explorer.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
