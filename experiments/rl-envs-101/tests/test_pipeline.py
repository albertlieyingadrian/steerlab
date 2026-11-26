"""CPU tests for step 4: logs -> reconstruction -> verifier -> Harbor folder, on one small simulated week."""

import json
import runpy
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envlab import harbor_pack
from envlab import pipeline as P
from envlab.desk import DAYS, TIMES, Task, call_tool, copy_db, verify, world_to_json
from envlab.logs import assemble, plan_weeks

DOCTOR_IDS = {name: f"D{i + 1}" for i, name in enumerate(sorted(["Dr Kim", "Dr Okafor", "Dr Patel", "Dr Rossi"]))}


@pytest.fixture(scope="module")
def week():
    sessions = plan_weeks(n_weeks=1, per_week=60, model_share=0.0, seed=3)
    logs, truth = assemble(sessions, {})
    return [P.parse_session(log) for log in logs], {t["session_id"]: t for t in truth}


def test_logs_keep_no_truth(week):
    sessions, _ = week
    assert set(sessions[0]) >= {"session_id", "started_at", "agent", "messages", "feedback"}
    assert "task" not in sessions[0] and "start_world" not in sessions[0]


def test_every_rebuilt_world_replays_its_log(week):
    sessions, _ = week
    for s in sessions:
        assert P.replay_matches(s, P.build_world(P.observed_start(s), DOCTOR_IDS, TIMES, DAYS)), s["session_id"]


def test_extracted_tasks_grade_like_the_truth(week):
    sessions, truth = week
    n = 0
    for s in sessions:
        kind = truth[s["session_id"]]["task"]["kind"]
        obs = P.observed_start(s)
        world = P.build_world(obs, DOCTOR_IDS, TIMES, DAYS)
        task, _ = P.extract_task(s, kind, obs, DOCTOR_IDS)
        if task is None:
            continue
        end = copy_db(world)
        for c in s["calls"]:
            if c["result"] is not None:
                call_tool(end, str(c["tool"]), c["args"])
        assert verify(world, end, task)["hidden"] == truth[s["session_id"]]["scores"]["hidden"]
        n += 1
    assert n > 40


def test_rule_derivation_prefers_earliest(week):
    sessions, truth = week
    kinds = [truth[s["session_id"]]["task"]["kind"] for s in sessions]
    assert P.choose_rule(P.slot_rule_evidence(sessions, kinds)) == "earliest"


def test_redact_removes_phones():
    text = "This is Ana Lopez (555-0101). I can't make it. You can reach me at 555-0101."
    assert P.private_data(P.redact(text)) == []


def test_harbor_folder_oracle_and_nop(tmp_path, week):
    sessions, truth = week
    s = next(x for x in sessions if truth[x["session_id"]]["task"]["kind"] == "reschedule")
    obs = P.observed_start(s)
    world = P.build_world(obs, DOCTOR_IDS, TIMES, DAYS)
    task, why = P.extract_task(s, "reschedule", obs, DOCTOR_IDS)
    assert task is not None, why
    folder = harbor_pack.write_task({"task": task.to_json(), "world": world_to_json(world)}, tmp_path)
    assert {p.name for p in (folder / "environment").iterdir()} == {
        "Dockerfile",
        "desk.py",
        "desk_cli.py",
        "world.json",
    }
    cli = runpy.run_path(str(folder / "environment" / "desk_cli.py"))
    for solved in (True, False):
        db = tmp_path / f"clinic_{solved}.db"
        cli["main"](["--init", str(db)])
        if solved:
            for line in (folder / "solution" / "solve.sh").read_text().splitlines():
                if line.startswith("desk "):
                    cli["main"](line.split()[1:], str(db))
        graded = Task(**json.loads((folder / "tests" / "task.json").read_text()))
        assert verify(world, sqlite3.connect(db), graded)["hidden"] == (1.0 if solved else 0.0)
