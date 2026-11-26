"""Step 4, step 0: make the logged trajectories the rest of the lesson starts from.

    python experiments/rl-envs-101/04a_make_logs.py plan        scripted sessions now; model sessions -> pending file
    MODAL_PROFILE=albertlieyingadrian modal run experiments/rl-envs-101/modal_app.py::rollouts_main \
        --items experiments/rl-envs-101/data/logs/pending_model.jsonl \
        --out experiments/rl-envs-101/data/logs/model_sessions.jsonl
    python experiments/rl-envs-101/04a_make_logs.py assemble    merge, then write desk_logs.jsonl and truth.jsonl

The first command is free and takes seconds; the Modal call runs about 120 sessions on one L40S (a few minutes).
"""

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envlab.logs import assemble, dump_jsonl, load_jsonl, plan_weeks, replay

LOGS = HERE / "data" / "logs"
PLAN = LOGS / "plan.pkl"  # scratch: the full plan, rebuilt by `plan` (deterministic)

if sys.argv[1:] == ["plan"]:
    sessions = plan_weeks()
    LOGS.mkdir(parents=True, exist_ok=True)
    PLAN.write_bytes(pickle.dumps(sessions))
    pending = [
        {"id": s["session_id"], "task": s["task"], "world": s["start_world"]}
        for s in sessions
        if s["status"] == "pending"
    ]
    dump_jsonl(pending, LOGS / "pending_model.jsonl")
    print(f"{len(sessions)} sessions: {len(sessions) - len(pending)} scripted, {len(pending)} pending for the model")
    print("scripted habits:", dict(Counter(s["habit"] for s in sessions if s["habit"])))
elif sys.argv[1:] == ["assemble"]:
    sessions = pickle.loads(PLAN.read_bytes())
    by_id = {s["session_id"]: s for s in sessions}
    results = {}
    for row in load_jsonl(LOGS / "model_sessions.jsonl"):
        again = replay(by_id[row["id"]], row["messages"])  # the end state must be reproducible from the chat alone
        assert all(again[k] == row["scores"][k] for k in ("reward", "hidden", "collateral")), row["id"]
        results[row["id"]] = {"messages": row["messages"], "scores": row["scores"]}
    logs, truth = assemble(sessions, results)
    dump_jsonl(logs, LOGS / "desk_logs.jsonl")
    dump_jsonl(truth, LOGS / "truth.jsonl")
    print(f"wrote {len(logs)} logs; feedback: {dict(Counter(r['feedback'] for r in logs))}")
    for agent in sorted({t["agent"] for t in truth}):
        rows = [t for t in truth if t["agent"] == agent]
        mean = {k: round(sum(t["scores"][k] for t in rows) / len(rows), 3) for k in ("reward", "hidden", "collateral")}
        print(f"  {agent:24s} n={len(rows):3d}  {json.dumps(mean)}")
else:
    raise SystemExit(__doc__)
