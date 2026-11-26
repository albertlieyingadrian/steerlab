"""Scripted front-desk agents: each one is a policy with a known habit, so a verifier can be tested on it.

    careful    does the task right: right patient (date of birth), earliest slot, frees the old slot
    sloppy     takes the first slot listed (list order is not time order), passes the visible check, often fails hidden
    confused   takes the first patient match even when two patients share a name
    greedy     books every free slot that day (the visible check passes; the calendar is wrecked)
    lazy       says it is done without touching anything (only cancel-style checks could ever pass)

A scripted agent reads the instruction with regular expressions, then talks to the environment in the same JSON
protocol a model would, so its trajectories look exactly like a model's.
"""

from __future__ import annotations

import json
import re

from envlab.desk import DOCTORS, DeskEnv, Task, patient_row

NAME_RE = re.compile(r"(?:Book|Cancel|Move) ([A-Z][a-z]+ [A-Z][a-z]+)(?: \(born (\d{4}-\d{2}-\d{2})\))?")
DOCTOR_RE = re.compile(r"(Dr [A-Z][a-z]+)")
DAY_RE = re.compile(r"\b(?:on|to) (Mon|Tue|Wed|Thu|Fri)\b")
HABITS = ["careful", "sloppy", "confused", "greedy", "lazy"]


def read_instruction(text: str) -> dict:
    name = NAME_RE.search(text)
    day = DAY_RE.search(text)
    kind = "book" if text.startswith("Book") else "cancel" if text.startswith("Cancel") else "reschedule"
    return {
        "kind": kind,
        "name": name.group(1) if name else "",
        "dob": name.group(2) if name else None,
        "doctor": DOCTOR_RE.search(text).group(1),
        "day": day.group(1) if day else None,
    }


def spec_from_task(task: Task) -> dict:
    """What a scripted agent needs, read from the structured task (not the text), so any wording works."""
    return {
        "kind": task.kind,
        "name": None,  # filled from the world by run_scripted
        "dob": None,
        "doctor": dict(DOCTORS)[task.doctor_id],
        "day": task.day,
        "patient_id": task.patient_id,
    }


def scripted_actions(habit: str, spec: dict):
    """A generator policy: send() it each observation, it yields the next action (a JSON string)."""
    if habit == "lazy":
        yield json.dumps({"done": "All set."})
        return
    found = json.loads((yield json.dumps({"tool": "find_patient", "args": {"name": spec["name"]}})))
    matches = found.get("matches", [])
    if not matches:
        yield json.dumps({"done": "I could not find that patient."})
        return
    patient = matches[0]
    if habit != "confused" and spec.get("patient_id"):
        patient = next((m for m in matches if m["patient_id"] == spec["patient_id"]), matches[0])
    elif habit != "confused" and spec["dob"]:
        patient = next((m for m in matches if m["dob"] == spec["dob"]), matches[0])
    old = next((a for a in patient["appointments"] if a["doctor"] == spec["doctor"]), None)
    if spec["kind"] != "cancel":
        args = {"doctor": spec["doctor"], "day": spec["day"]}
        listed = json.loads((yield json.dumps({"tool": "list_slots", "args": args})))
        free = listed.get("free", [])
        if not free:
            yield json.dumps({"done": "No free slots that day."})
            return
        if habit == "greedy":
            chosen = free
        elif habit == "sloppy":
            chosen = free[:1]
        else:
            chosen = [min(free, key=lambda s: s["time"])]
        for slot in chosen:
            yield json.dumps(
                {"tool": "book", "args": {"patient_id": patient["patient_id"], "slot_id": slot["slot_id"]}}
            )
    if old and spec["kind"] in ("cancel", "reschedule"):
        yield json.dumps({"tool": "cancel", "args": {"slot_id": old["slot_id"]}})
    yield json.dumps({"done": "Done, your calendar is updated."})


def run_scripted(env: DeskEnv, task: Task, habit: str) -> dict:
    """One episode with a scripted habit. Returns the trajectory as chat messages plus the verifier's keys."""
    observation = env.reset(task)
    messages = [{"role": "user", "content": observation}]
    spec = spec_from_task(task)
    spec["name"], spec["dob"] = patient_row(env.world, task.patient_id)
    policy = scripted_actions(habit, spec)
    action = next(policy)
    while True:
        messages.append({"role": "assistant", "content": action})
        observation, _, done, info = env.step(action)
        if done:
            return {"messages": messages, "scores": info}
        messages.append({"role": "tool", "content": observation})
        try:
            action = policy.send(observation)
        except StopIteration:
            action = json.dumps({"done": "Done."})
