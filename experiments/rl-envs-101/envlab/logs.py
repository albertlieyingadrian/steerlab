"""Step 4, step 0: make our own "production logs" of a front-desk assistant, with no third-party data.

Three simulated clinic weeks. Each week starts from a fresh seeded world and runs 160 sessions in order, so the
calendar changes as the week goes on. A session is one patient request, worded in one of a few ways (some mention a
phone number, as real requests do), handled by one of two agents:

    "desk-bot-v2"              a scripted agent with a habit (careful 60%, sloppy 20%, confused 10%, greedy 5%, lazy 5%)
    "qwen3-4b-instruct-2507"   an open model run on Modal through the same tools (a quarter of the sessions)

A log keeps only what a real system would keep: the chat (request, tool calls, tool results, replies), the agent
name, a timestamp and the patient's feedback ("resolved", "not_resolved", or missing). The true task, the habit and
the starting world go to a separate truth file, used only to grade the pipeline that rebuilds them.
"""

from __future__ import annotations

import datetime as dt
import json
import random

from envlab.agents import run_scripted
from envlab.desk import DeskEnv, Task, copy_db, make_tasks, make_world, patient_row, world_from_json, world_to_json

DAY_NAMES = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}
WORDINGS = {
    "book": [
        "Book {who} with {doctor} on {day}, at the earliest free time.",
        "Hi, I'm {who}. Can I get the first opening with {doctor} on {dayname}? You can reach me at {phone}.",
        "{who} needs to see {doctor} on {dayname}, whatever is earliest please.",
        "Please schedule {who} with {doctor}, {dayname}, earliest slot available.",
    ],
    "reschedule": [
        "Move {who}'s appointment with {doctor} to {day}, at the earliest free time, and free the old slot.",
        "This is {who} ({phone}). I can't make my {doctor} appointment, can you move it to {dayname}, as early as possible?",
        "Reschedule {who} with {doctor} to {dayname}, earliest time, and cancel the old one.",
    ],
    "cancel": [
        "Cancel {who}'s appointment with {doctor}.",
        "Hi, {who} here, please cancel my appointment with {doctor}. Call {phone} if needed.",
        "{who} won't make it to see {doctor}; please cancel the booking.",
    ],
}
HABIT_MIX = [("careful", 0.6), ("sloppy", 0.2), ("confused", 0.1), ("greedy", 0.05), ("lazy", 0.05)]
MODEL_AGENT = "qwen3-4b-instruct-2507"
SCRIPTED_AGENT = "desk-bot-v2"


def word_task(db, task: Task, rng: random.Random) -> str:
    name, dob = patient_row(db, task.patient_id)
    phone = db.execute("SELECT phone FROM patients WHERE id=?", (task.patient_id,)).fetchone()[0]
    shared = db.execute("SELECT count(*) FROM patients WHERE name=?", (name,)).fetchone()[0] > 1
    who = f"{name} (born {dob})" if shared else name
    doctor = db.execute("SELECT name FROM doctors WHERE id=?", (task.doctor_id,)).fetchone()[0]
    template = rng.choice(WORDINGS[task.kind])
    return template.format(who=who, doctor=doctor, day=task.day, dayname=DAY_NAMES.get(task.day or "", ""), phone=phone)


def plan_weeks(n_weeks: int = 3, per_week: int = 160, model_share: float = 0.25, seed: int = 0) -> list[dict]:
    """Run every scripted session; return all sessions, with model sessions marked "pending" for Modal."""
    rng = random.Random(seed)
    sessions = []
    for week in range(1, n_weeks + 1):
        world = make_world(seed=100 + week)
        start = dt.datetime(2026, 3, 2) + dt.timedelta(days=7 * (week - 1))
        for i in range(per_week):
            kind = rng.choices(["book", "reschedule", "cancel"], [0.4, 0.3, 0.3])[0]
            task = make_tasks(world, 1, seed=rng.randrange(10**9), prefix=f"w{week}-{i:03d}-", kinds=(kind,))[0]
            task.instruction = word_task(world, task, rng)
            when = start + dt.timedelta(days=i // 32, minutes=8 * 60 + (i % 32) * 15)
            base = {
                "session_id": f"w{week}-{i:03d}",
                "week": week,
                "started_at": when.isoformat(timespec="minutes"),
                "task": task.to_json(),
                "start_world": world_to_json(world),
            }
            env = DeskEnv(copy_db(world), max_turns=40)
            if rng.random() < model_share:
                sessions.append({**base, "agent": MODEL_AGENT, "habit": None, "status": "pending"})
                run_scripted(env, task, "careful")  # the calendar moves on as a careful agent would have left it
            else:
                habit = rng.choices([h for h, _ in HABIT_MIX], [w for _, w in HABIT_MIX])[0]
                episode = run_scripted(env, task, habit)
                sessions.append({**base, "agent": SCRIPTED_AGENT, "habit": habit, "status": "done", **episode})
            world = env.db
    return sessions


def feedback(hidden: float, rng: random.Random) -> str | None:
    """What the patient said afterwards: missing 15% of the time, and wrong 5% of the time (people misread)."""
    if rng.random() < 0.15:
        return None
    resolved = hidden == 1.0
    if rng.random() < 0.05:
        resolved = not resolved
    return "resolved" if resolved else "not_resolved"


def assemble(sessions: list[dict], model_results: dict[str, dict], seed: int = 1) -> tuple[list[dict], list[dict]]:
    """Merge the model sessions in, then split every session into a log row and a truth row."""
    rng = random.Random(seed)
    logs, truth = [], []
    for s in sessions:
        if s["status"] == "pending":
            s = {**s, **model_results[s["session_id"]]}
        scores = s["scores"]
        logs.append(
            {
                "session_id": s["session_id"],
                "started_at": s["started_at"],
                "agent": s["agent"],
                "messages": s["messages"],
                "feedback": feedback(scores["hidden"], rng),
            }
        )
        truth.append(
            {
                "session_id": s["session_id"],
                "week": s["week"],
                "task": s["task"],
                "habit": s["habit"],
                "agent": s["agent"],
                "scores": {k: scores[k] for k in ("reward", "hidden", "collateral")},
                "start_world": s["start_world"],
            }
        )
    return logs, truth


def replay(session: dict, messages: list[dict]) -> dict:
    """Replay a session's assistant turns on its start world and grade the end state."""
    task = Task(**session["task"])
    env = DeskEnv(world_from_json(session["start_world"]), max_turns=10**6)
    env.reset(task)
    for m in messages:
        if m["role"] == "assistant":
            _, _, done, info = env.step(m["content"])
            if done:
                return info
    return env._finish("out of turns")[3]


def dump_jsonl(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
