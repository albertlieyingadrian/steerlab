"""Turn a folder of OpenEnv result JSONs (one per task) into one compact row per task.

    .venv/bin/python experiments/agent-traces/rollout_table.py data/released_rollouts/eval-sft-500 --run released-sft-500 \
        --split eval --out results/per_task/released-sft-500.eval.jsonl

A row: run, split, index (the task's position in the split, from the file name), task_name, solved (the grader's
correctness, or reward on graders that write only that, equals 1), graded, n_turns, n_tool_calls, wall_s,
answer_written (a tool call that writes answer.txt), finished (the last agent turn ended without a tool call),
tool_names, exception (OpenEnv's exception type) and timeout (the agent hit the task's time limit). A task in the split with no result file counts as unsolved; `data/splits/` fixes the denominators.
"""

import argparse
import json
import re
from pathlib import Path


def task_reward(result: dict) -> float | None:
    rewards = result.get("rewards") or {}
    for value in (rewards.get("correctness"), result.get("reward"), rewards.get("reward")):
        if value is not None:
            return value
    return None


def tool_calls(turn: dict) -> list[dict]:
    calls = turn.get("tool_calls") or []
    return calls if isinstance(calls, list) else []


ANSWER = re.compile(r"answer\.txt")


def row(path: Path, run: str, split: str) -> dict:
    data = json.loads(path.read_text())
    data = data[0] if isinstance(data, list) else data
    turns = [t for t in data.get("turns", []) if t.get("role") == "agent" and not t.get("discarded")]
    calls = [c for t in turns for c in tool_calls(t)]
    reward = task_reward(data)
    return {
        "run": run,
        "split": split,
        "index": int(path.stem.split("_")[-1]),
        "task_name": data.get("task_name"),
        "solved": int(reward == 1.0),
        "graded": reward is not None,
        "n_turns": len(turns),
        "n_tool_calls": len(calls),
        "wall_s": round(data.get("wall_s") or 0, 1),
        "answer_written": any(ANSWER.search(str(c.get("arguments", ""))) for c in calls),
        "finished": bool(turns) and turns[-1].get("finish_reason") != "tool_calls",
        "tool_names": sorted({c.get("name", "") for c in calls}),
        "exception": data.get("exception_type") or "",
        "timeout": data.get("exception_type") == "AgentTimeoutError",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--run", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", default=None, help="run log: adds tasks whose result JSON never reached the Hub")
    a = p.parse_args()
    rows = [row(f, a.run, a.split) for f in sorted(Path(a.folder).glob("result_*.json"))]
    n_json = len(rows)
    if a.log:
        rows += log_rows(Path(a.log), a.run, a.split, {r["index"] for r in rows})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["index"])))
    print(
        f"{a.run} {a.split}: {n_json} result files + {len(rows) - n_json} from the log, "
        f"{sum(r['solved'] for r in rows)} solved -> {a.out}"
    )


LOG_LINE = re.compile(r"^\[(?P<name>[^\]]+)\] reward=(?P<reward>\S+) turns=(?P<turns>\S+) (?P<wall>[\d.]+)s")


def log_rows(log: Path, run: str, split: str, have: set[int]) -> list[dict]:
    """Rows for tasks the run log graded but whose result JSON is missing: the reward line rollouts.py prints after
    each task (`[task] reward=1.0 turns=7 131s`). Only solved, n_turns and wall_s are known; the rest are None."""
    splits = Path(__file__).resolve().parent / "data" / "splits"
    if split == "eval":
        index = {n: i for i, n in enumerate(splits.joinpath("eval_tasks.txt").read_text().split())}
    else:
        idx = [int(x) for x in splits.joinpath("test_indices.txt").read_text().strip().split(",")]
        index = dict(zip(splits.joinpath("test_tasks.txt").read_text().split(), idx))
    out, seen = [], set()
    for line in log.read_text().splitlines():
        m = LOG_LINE.match(line.strip())
        if not m or m["name"] not in index or index[m["name"]] in have | seen:
            continue
        reward = None if m["reward"] == "None" else float(m["reward"])
        seen.add(index[m["name"]])
        out.append({"run": run, "split": split, "index": index[m["name"]], "task_name": m["name"],
                    "solved": int(reward == 1.0), "graded": reward is not None,
                    "n_turns": None if m["turns"] == "None" else int(m["turns"]), "n_tool_calls": None,
                    "wall_s": float(m["wall"]), "answer_written": None, "finished": None, "tool_names": None,
                    "exception": None, "timeout": None, "from_log": True})  # fmt: skip
    return out


if __name__ == "__main__":
    main()
