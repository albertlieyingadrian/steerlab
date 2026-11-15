"""Archive every trial of a run_seq.sh / run_par.sh batch and write one row per trial, so grading and analysis need
no API key.

    python3 collect_trials.py <label> [--port PATH]

Each trial's artifacts are found exactly, through AgentEnv's own records of the trial's instance (its document
store): the grader's result.json (task checks and trycua's A22 criteria), the Claude Code trajectory (every model
reply, tool call and usage) and the env snapshot files (the sentinel's log, the apps' state). Two trials can run at
once, so time windows are used only for old logs that name no instance. Everything is copied to <label>/raw/<n>/ and
<label>/trials.csv gets the outcome, the A22 criteria (background trials), turns and cost at list price.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

RUNS = Path(os.environ.get("CUA_RUNS", Path.home() / "oss/cua-runs"))
STATE = Path.home() / ".local/state/agent-env"
STORE = STATE / "object_store"
DB = STATE / "document_store/local.db"
TRAJ = STORE / "prompt_agent_trajectories/prompt_id=task"
SNAPS = STORE / "agentenv-snapshots"
#: $ per M tokens: input, cache write, cache read, output.
PRICES = {"opus": (4.0, 5.0, 0.40, 20.0), "haiku": (1.0, 1.25, 0.10, 5.0), "fable": (4.0, 5.0, 0.40, 20.0)}
A22 = ["key_focus", "window_raised", "typing", "pointer", "clipboard", "space", "dialog", "activation"]
TOKENS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def a22_from_log(sentinel_dir: Path, port: Path) -> dict:
    """trycua's A22 criteria from a sentinel log, computed as the port's grader does (for trials with no record)."""
    art = port / "artifacts"
    sys.path.insert(0, str(art / "grader"))
    grade = _module(art / "grader/grade.py")
    summ, v2 = _module(art / "taskpack/scoring/summarize_sentinel.py"), _module(art / "taskpack/scoring/bench_v2.py")
    log = sentinel_dir / "sentinel.jsonl"
    lines = grade._lines(log)
    start = max((i for i, line in enumerate(lines) if line["ev"] == "start"), default=0)
    lines = lines[start:]
    s = summ.summarize(str(log))
    window = grade._armed_window(lines)
    armed = next((line for line in window if line["ev"] == "armed"), {})
    m = v2.multiplayer_score(s, bool(s["available"] and grade._armed_in_front(armed)), True, "background")
    return {"undisturbed": m["user_undisturbed"], **m["criteria"]}


def usage(path: Path) -> dict:
    events = json.loads(path.read_text())
    events = events if isinstance(events, list) else events.get("events") or []
    model = next((e.get("model", "") for e in events if isinstance(e, dict) and e.get("type") == "system"), "")
    results = [e for e in events if isinstance(e, dict) and e.get("type") == "result"]
    r = results[-1] if results else {}
    u = r.get("usage") or {}
    if not u:  # cut off (turn or time limit) before Claude Code's result: add up each model reply's usage once
        seen: dict[str, dict] = {}
        for e in events:
            msg = e.get("message") if isinstance(e, dict) and e.get("type") == "assistant" else None
            if isinstance(msg, dict) and msg.get("usage"):
                seen[msg.get("id") or str(len(seen))] = msg["usage"]
        u = {k: sum(v.get(k, 0) or 0 for v in seen.values()) for k in TOKENS}
        r = {"num_turns": len(seen)}
    key = next((k for k in PRICES if k in model), "opus")
    pi, pw, pr, po = PRICES[key]
    cost = (
        u.get("input_tokens", 0) * pi
        + u.get("cache_creation_input_tokens", 0) * pw
        + u.get("cache_read_input_tokens", 0) * pr
        + u.get("output_tokens", 0) * po
    ) / 1e6
    return {
        "model": model,
        "turns": r.get("num_turns", ""),
        "cost": round(cost, 4),
        "reported": r.get("total_cost_usd", ""),
        **{k: u.get(k, 0) for k in TOKENS},
    }


def journal(con: sqlite3.Connection, inst: str) -> dict:
    """A trial instance's grade result, its trajectory ids and its snapshot files, from AgentEnv's records."""
    out: dict = {"result": None, "trajectories": [], "snapshot_files": {}}
    conv_ids = []
    for (doc,) in con.execute(
        "select doc from docs_task_step_journal where json_extract(doc,'$.instance_id')=?", (inst,)
    ):
        for op in json.loads(doc).get("ops", {}).get("sets", []):
            if op["path"].endswith("grade.extracted_files"):
                out["result"] = (op["value"] or {}).get("/grade/result.json")
            if op["path"].startswith("context.metadata.a2a_conversations."):
                conv_ids.append(op["value"])
    for cid in conv_ids:
        for (doc,) in con.execute(
            "select doc from docs_agent_env_a2a_conversations where json_extract(doc,'$.conversation_id')=?", (cid,)
        ):
            out["trajectories"] += [t["a2a_task_id"] for t in json.loads(doc).get("a2a_tasks", [])]
    for (doc,) in con.execute(
        "select doc from docs_artifacts where json_extract(doc,'$.id') like ? and json_extract(doc,'$.type')='file'",
        (inst + "__snapshot-snapshot%",),
    ):
        url = json.loads(doc).get("object_url") or ""
        if url.startswith("file://"):
            path = Path(url[len("file://") :])
            rel = path.relative_to(SNAPS) if SNAPS in path.parents else Path(path.name)
            out["snapshot_files"][str(Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel)] = path
    return out


def outcome(line: str) -> str:
    if "passed (" in line:
        return "pass"
    if "scored below 1" in line:
        return "fail"
    if "Reached maximum number of turns" in line:
        return "fail_turns"
    if "timed out" in line:
        return "fail_time"
    if "failed at step 'task'" in line:
        return "fail_agent"
    return "harness" if line else "no_result"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument(
        "--port", type=Path, default=Path(os.environ.get("PORT", Path.home() / "oss/agentenv-cua-driver-bench"))
    )
    args = p.parse_args()
    run = RUNS / args.label
    done = [line.split("\t") for line in (run / "done.txt").read_text().splitlines() if line.strip()]
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    trajs = list(TRAJ.glob("trajectory-*.json"))
    runs_seen: dict[tuple, int] = {}
    rows = []
    for n, task in sorted(done):
        logs = sorted(
            (q for q in run.glob(f"{n}-{task}*.log") if "driverpy-error" not in q.name), key=lambda q: q.stat().st_mtime
        )
        if not logs:
            continue
        t0, t1 = logs[-1].stat().st_birthtime, logs[-1].stat().st_mtime + 5
        text = logs[-1].read_text(errors="replace")
        line = next((ln.strip() for ln in reversed(text.splitlines()) if ln.startswith(f"  tasks/{task}.json")), "")
        arm = "ourdriver" if "-ourdriver-" in task else "cua-driver" if "-cua-driver-" in task else "?"
        cond = "background" if task.endswith("background") else "foreground"
        variant = re.sub(r"-r\d-", "-", re.sub(r"^macos-(ourdriver|cua-driver)-", "", task))
        key = (arm, variant)
        runs_seen[key] = runs_seen.get(key, 0) + 1
        raw = run / "raw" / n
        raw.mkdir(parents=True, exist_ok=True)
        insts = re.findall(r"instance (@local/[^\s,]+)", text)  # the summary line wraps, so search the whole log
        crit: dict = {}
        if insts:
            j = journal(con, insts[-1])
            mine_t = [TRAJ / f"trajectory-{t}.json" for t in j["trajectories"]]
            mine_t = [q for q in mine_t if q.exists()]
            for q in mine_t:
                if not (raw / q.name).exists():
                    shutil.copy2(q, raw / q.name)
            for rel, src in j["snapshot_files"].items():
                dst = raw / "snapshot" / rel
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            if j["result"] is not None:
                (raw / "grade_result.json").write_text(json.dumps(j["result"], indent=1) + "\n")
                und = next((r for r in j["result"].get("results", []) if r.get("id") == "user_undisturbed"), None)
                if und:
                    crit = {"undisturbed": und["result"], **und.get("criteria", {})}
            (raw / "instance.txt").write_text(insts[-1] + "\n")
        else:  # an old log with no instance named: fall back to the time window
            mine_t = sorted((q for q in trajs if t0 <= q.stat().st_mtime <= t1), key=lambda q: q.stat().st_mtime)
            if cond == "background":
                snaps = [d for d in SNAPS.iterdir() if d.is_dir() and t0 <= d.stat().st_mtime <= t1]
                sent = next(
                    (d / ".bench-sentinel" for d in snaps if (d / ".bench-sentinel/sentinel.jsonl").exists()), None
                )
                if sent:
                    try:
                        crit = a22_from_log(sent, args.port)
                    except Exception as e:
                        crit = {"undisturbed": f"error: {e}"}
        u = usage(mine_t[-1]) if mine_t else {}
        grade_m, gui_m = re.search(r"grade: (\d)", text[-3000:]), re.search(r"gui_only: (\d)", line)
        rows.append(
            {
                "n": n,
                "task": task,
                "arm": arm,
                "variant": variant,
                "condition": cond,
                "run": runs_seen[key],
                "outcome": outcome(line),
                "passed": int(outcome(line) == "pass"),
                "grade": grade_m.group(1) if grade_m else "",
                "gui_only": gui_m.group(1) if gui_m else "",
                "attempts": len(logs),
                "seconds": round(t1 - 5 - t0),
                **{k: u.get(k, "") for k in ("model", "turns", "cost", "reported", *TOKENS)},
                "undisturbed": crit.get("undisturbed", ""),
                **{c: crit.get(c, "") for c in A22},
                "trajectory": mine_t[-1].name if mine_t else "",
                "instance": insts[-1].rsplit("/", 1)[-1] if insts else "",
                "result": line.split(", instance")[0][:300],
            }
        )
    con.close()
    with (run / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    for arm, rs in by.items():
        cost = sum(float(r["cost"] or 0) for r in rs)
        print(
            f"{arm}: {sum(r['passed'] for r in rs)}/{len(rs)} passed, "
            f"harness {sum(r['outcome'] == 'harness' for r in rs)}, ${cost:.2f}"
        )


if __name__ == "__main__":
    main()
