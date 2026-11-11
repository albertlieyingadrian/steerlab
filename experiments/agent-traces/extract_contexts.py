"""Pull whole agent conversations out of OpenEnv result JSONs, for the interpretability runs.

    .venv/bin/python experiments/agent-traces/extract_contexts.py data/released_rollouts/eval-teacher-397b \
        --out data/contexts/teacher.eval.jsonl

One row per task: index, task_name, solved, and messages = the agent's last request (system prompt, task, every
assistant turn with its tool calls, every tool output) plus its final reply, with tool-call arguments parsed to dicts
(as the guide's build_dataset.py does, since the Qwen3.5 template iterates over them). The tool schemas are the same
for every task and are written once, to data/contexts/tools.json.
"""

import argparse
import json
from pathlib import Path

from rollout_table import task_reward


def parse_calls(calls: list[dict]) -> list[dict]:
    out = []
    for c in calls or []:
        fn = c.get("function", c)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append({"type": "function", "function": {"name": fn.get("name"), "arguments": args}})
    return out


def conversation(data: dict) -> tuple[list[dict], list[dict]] | None:
    turns = [t for t in data.get("turns", []) if t.get("role") == "agent" and not t.get("discarded")]
    if not turns:
        return None
    last = turns[-1]
    msgs = []
    for m in last["request_messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = parse_calls(m["tool_calls"])
        if not isinstance(m.get("content"), str):
            m["content"] = "".join(p.get("text", "") for p in m.get("content") or [] if isinstance(p, dict))
        msgs.append(m)
    final = {"role": "assistant", "content": last.get("text") or ""}
    if last.get("tool_calls"):
        final["tool_calls"] = parse_calls(last["tool_calls"])
    return msgs + [final], last.get("request_tools") or []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, tools = [], None
    for f in sorted(Path(a.folder).glob("result_*.json")):
        data = json.loads(f.read_text())
        data = data[0] if isinstance(data, list) else data
        conv = conversation(data)
        if conv is None:
            continue
        msgs, tools_here = conv
        tools = tools or tools_here
        rows.append({"index": int(f.stem.split("_")[-1]), "task_name": data.get("task_name"),
                     "solved": int(task_reward(data) == 1.0), "messages": msgs})  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tools_path = out.parent / "tools.json"
    if tools and not tools_path.exists():
        tools_path.write_text(json.dumps(tools, indent=1))
    print(f"{len(rows)} conversations -> {out}")


if __name__ == "__main__":
    main()
