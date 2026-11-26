"""Step 3: a multi-turn tool-use environment, a clinic front desk.

The world is a small SQLite database (patients, doctors, appointment slots). The agent changes it only through four
tools, and the verifier never reads what the agent *says*: it compares the database at the end with the database the
task should have produced. That is the difference between grading an essay ("I booked it!") and checking the
calendar.

    find_patient(name)            -> matching patients, with date of birth and current appointments
    list_slots(doctor, day)       -> that doctor's free slots that day (in slot-id order, NOT time order)
    book(patient_id, slot_id)     -> put the patient in a free slot
    cancel(slot_id)               -> free a booked slot

Two verifier keys matter:

    reward  (visible) a loose check the agent could read in a test file: "the patient has an appointment with that
                      doctor on that day". Cheap, and hackable.
    hidden            the whole calendar equals the expected one: the *earliest* free slot, the right patient (two
                      patients share a name), and nothing else touched.

Everything is synthetic: names are drawn from two short lists, phone numbers are 555-01xx (reserved for fiction).
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from dataclasses import asdict, dataclass

DOCTORS = [("D1", "Dr Kim"), ("D2", "Dr Patel"), ("D3", "Dr Okafor"), ("D4", "Dr Rossi")]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
TIMES = [f"{h:02d}:00" for h in range(9, 17)]
FIRST = ["Ana", "Ben", "Chloe", "Dev", "Elif", "Farah", "Goran", "Hana", "Ivan", "Jade", "Kofi", "Lena"]
LAST = ["Lopez", "Novak", "Singh", "Mbeki", "Tanaka", "Silva", "Haddad", "Berg"]
SCHEMA = """
CREATE TABLE patients (id TEXT PRIMARY KEY, name TEXT, dob TEXT, phone TEXT);
CREATE TABLE doctors (id TEXT PRIMARY KEY, name TEXT);
CREATE TABLE slots (id TEXT PRIMARY KEY, doctor_id TEXT, day TEXT, time TEXT, patient_id TEXT);
"""

# --- the world -------------------------------------------------------------------------------------------------------


def make_world(seed: int, n_patients: int = 24, booked_share: float = 0.45) -> sqlite3.Connection:
    """A seeded clinic week: 4 doctors x 5 days x 8 hours = 160 slots, about 45% already booked.

    Slot ids are shuffled, so listing slots by id does not list them by time (the trap in "earliest free slot").
    Two pairs of patients share a name and differ only by date of birth.
    """
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    names = rng.sample([f"{first} {last}" for first in FIRST for last in LAST], n_patients - 2)
    names += rng.sample(names, 2)  # two shared names
    for i, name in enumerate(names):
        dob = f"{rng.randint(1950, 2015)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        db.execute("INSERT INTO patients VALUES (?,?,?,?)", (f"P{i + 1:03d}", name, dob, f"555-01{i:02d}"))
    db.executemany("INSERT INTO doctors VALUES (?,?)", DOCTORS)
    cells = [(d, day, t) for d, _ in DOCTORS for day in DAYS for t in TIMES]
    ids = rng.sample(range(100, 1000), len(cells))
    patient_ids = [f"P{i + 1:03d}" for i in range(n_patients)]
    for (doctor, day, time), sid in zip(cells, ids):
        patient = rng.choice(patient_ids) if rng.random() < booked_share else None
        if (
            patient
            and db.execute("SELECT 1 FROM slots WHERE doctor_id=? AND patient_id=?", (doctor, patient)).fetchone()
        ):
            patient = None  # at most one appointment per patient per doctor
        db.execute("INSERT INTO slots VALUES (?,?,?,?,?)", (f"S{sid}", doctor, day, time, patient))
    db.commit()
    return db


def copy_db(db: sqlite3.Connection) -> sqlite3.Connection:
    new = sqlite3.connect(":memory:")
    db.backup(new)
    return new


def world_to_json(db: sqlite3.Connection) -> dict:
    """A whole world as plain lists, for task packs and Harbor environments."""
    return {
        t: [list(r) for r in db.execute(f"SELECT * FROM {t} ORDER BY id")] for t in ("patients", "doctors", "slots")
    }


def world_from_json(data: dict) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    for table, rows in data.items():
        if rows:
            db.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)
    db.commit()
    return db


def slot_table(db: sqlite3.Connection) -> dict[str, str | None]:
    """The part of the world tasks change: slot id -> patient id (or None when free)."""
    return dict(db.execute("SELECT id, patient_id FROM slots ORDER BY id").fetchall())


def doctor_id(db: sqlite3.Connection, doctor: str) -> str | None:
    key = doctor.lower().replace("dr.", "").replace("dr ", "").strip()
    for did, name in db.execute("SELECT id, name FROM doctors"):
        if key in (did.lower(), name.lower(), name.lower().replace("dr ", "")):
            return did
    return None


def doctor_name(db: sqlite3.Connection, did: str) -> str:
    return db.execute("SELECT name FROM doctors WHERE id=?", (did,)).fetchone()[0]


# --- the four tools ----------------------------------------------------------------------------------------------------


def find_patient(db: sqlite3.Connection, name: str) -> dict:
    rows = db.execute("SELECT id, name, dob, phone FROM patients WHERE lower(name)=lower(?)", (name.strip(),))
    matches = []
    for pid, pname, dob, phone in rows.fetchall():
        appts = db.execute(
            "SELECT s.id, d.name, s.day, s.time FROM slots s JOIN doctors d ON d.id=s.doctor_id WHERE s.patient_id=? "
            "ORDER BY s.id",
            (pid,),
        ).fetchall()
        matches.append(
            {
                "patient_id": pid,
                "name": pname,
                "dob": dob,
                "phone": phone,
                "appointments": [{"slot_id": s, "doctor": d, "day": day, "time": t} for s, d, day, t in appts],
            }
        )
    return {"matches": matches}


def list_slots(db: sqlite3.Connection, doctor: str, day: str) -> dict:
    did = doctor_id(db, doctor)
    if did is None:
        return {"error": f"unknown doctor {doctor!r}"}
    day = day.strip()[:3].title()
    if day not in DAYS:
        return {"error": f"unknown day {day!r}; use Mon-Fri"}
    rows = db.execute(
        "SELECT id, time FROM slots WHERE doctor_id=? AND day=? AND patient_id IS NULL ORDER BY id", (did, day)
    ).fetchall()
    return {"doctor": doctor_name(db, did), "day": day, "free": [{"slot_id": s, "time": t} for s, t in rows]}


def book(db: sqlite3.Connection, patient_id: str, slot_id: str) -> dict:
    if not db.execute("SELECT 1 FROM patients WHERE id=?", (patient_id,)).fetchone():
        return {"ok": False, "error": f"unknown patient {patient_id!r}"}
    row = db.execute("SELECT doctor_id, day, time, patient_id FROM slots WHERE id=?", (slot_id,)).fetchone()
    if row is None:
        return {"ok": False, "error": f"unknown slot {slot_id!r}"}
    if row[3] is not None:
        return {"ok": False, "error": "slot is already booked"}
    db.execute("UPDATE slots SET patient_id=? WHERE id=?", (patient_id, slot_id))
    db.commit()
    return {"ok": True, "slot_id": slot_id, "doctor": doctor_name(db, row[0]), "day": row[1], "time": row[2]}


def cancel(db: sqlite3.Connection, slot_id: str) -> dict:
    row = db.execute("SELECT patient_id FROM slots WHERE id=?", (slot_id,)).fetchone()
    if row is None or row[0] is None:
        return {"ok": False, "error": "no booking in that slot"}
    db.execute("UPDATE slots SET patient_id=NULL WHERE id=?", (slot_id,))
    db.commit()
    return {"ok": True, "slot_id": slot_id}


TOOLS = {"find_patient": find_patient, "list_slots": list_slots, "book": book, "cancel": cancel}


def call_tool(db: sqlite3.Connection, name: str, args: dict) -> dict:
    """Run one tool call; bad names or arguments come back as an error the agent can read, never as a crash."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}; tools: {sorted(TOOLS)}"}
    try:
        return TOOLS[name](db, **args)
    except (TypeError, AttributeError) as exc:
        return {"error": f"bad arguments for {name}: {exc}"}


# --- tasks and the verifier ------------------------------------------------------------------------------------------


@dataclass
class Task:
    task_id: str
    kind: str  # "book" | "reschedule" | "cancel"
    instruction: str
    patient_id: str
    doctor_id: str
    day: str | None = None  # target day for book / reschedule
    old_slot_id: str | None = None  # the appointment to cancel or move

    def to_json(self) -> dict:
        return asdict(self)


def patient_row(db: sqlite3.Connection, pid: str) -> tuple[str, str]:
    return db.execute("SELECT name, dob FROM patients WHERE id=?", (pid,)).fetchone()


def earliest_free(db: sqlite3.Connection, did: str, day: str) -> str | None:
    row = db.execute(
        "SELECT id FROM slots WHERE doctor_id=? AND day=? AND patient_id IS NULL ORDER BY time LIMIT 1", (did, day)
    ).fetchone()
    return row[0] if row else None


def expected_slots(start: sqlite3.Connection, task: Task) -> dict[str, str | None]:
    """The calendar the task should end with, computed from the starting world."""
    table = slot_table(start)
    if task.old_slot_id:
        table[task.old_slot_id] = None
    if task.kind in ("book", "reschedule"):
        table[earliest_free(start, task.doctor_id, task.day)] = task.patient_id
    return table


def verify(start: sqlite3.Connection, final: sqlite3.Connection, task: Task) -> dict[str, float]:
    """Grade the final world. Always returns every key (a missing key would read as 0 in an average)."""
    final_table, expected = slot_table(final), expected_slots(start, task)
    rows = final.execute("SELECT doctor_id, day FROM slots WHERE patient_id=?", (task.patient_id,)).fetchall()
    with_doctor = [day for did, day in rows if did == task.doctor_id]
    visible = (not with_doctor) if task.kind == "cancel" else (task.day in with_doctor)
    collateral = sum(final_table.get(s) != expected[s] for s in expected)
    return {"reward": float(visible), "hidden": float(collateral == 0), "collateral": float(collateral)}


def make_tasks(
    db: sqlite3.Connection,
    n: int,
    seed: int,
    prefix: str = "t",
    kinds: tuple[str, ...] = ("book", "reschedule", "cancel"),
) -> list[Task]:
    """Draw n solvable tasks from a world, cycling through `kinds` (by default a third each)."""
    rng = random.Random(seed)
    attempts = 0
    tasks: list[Task] = []
    patients = [r[0] for r in db.execute("SELECT id FROM patients ORDER BY id")]
    shared = {r[0] for r in db.execute("SELECT name FROM patients GROUP BY name HAVING count(*) > 1")}
    while len(tasks) < n:
        attempts += 1
        if attempts > 1000 * n:
            raise ValueError("this world has no solvable task of the requested kinds")
        kind = kinds[len(tasks) % len(kinds)]
        pid = rng.choice(patients)
        name, dob = patient_row(db, pid)
        who = f"{name} (born {dob})" if name in shared else name
        booked = db.execute("SELECT id, doctor_id, day FROM slots WHERE patient_id=?", (pid,)).fetchall()
        booked_doctors = {d for _, d, _ in booked}
        tid = f"{prefix}{len(tasks):03d}"
        if kind == "book":
            free_doctors = [d for d, _ in DOCTORS if d not in booked_doctors]
            if not free_doctors:
                continue
            did, day = rng.choice(free_doctors), rng.choice(DAYS)
            if earliest_free(db, did, day) is None:
                continue
            text = f"Book {who} with {dict(DOCTORS)[did]} on {day}, at the earliest free time."
            task = Task(tid, kind, text, pid, did, day)
        else:
            if not booked:
                continue
            sid, did, old_day = rng.choice(booked)
            if sum(d == did for _, d, _ in booked) > 1:
                continue  # two appointments with one doctor: "cancel the appointment" would be ambiguous
            if kind == "cancel":
                task = Task(tid, kind, f"Cancel {who}'s appointment with {dict(DOCTORS)[did]}.", pid, did, None, sid)
            else:
                day = rng.choice([d for d in DAYS if d != old_day])
                if earliest_free(db, did, day) is None:
                    continue
                text = (
                    f"Move {who}'s appointment with {dict(DOCTORS)[did]} to {day}, at the earliest free time, "
                    "and free the old slot."
                )
                task = Task(tid, kind, text, pid, did, day, sid)
        tasks.append(task)
    return tasks


# --- the environment: reset / step over a JSON tool-call protocol ----------------------------------------------------

SYSTEM_PROMPT = """You are the front-desk assistant of a small clinic. You change the clinic's calendar only through tools.
Each turn, reply with exactly one JSON object and nothing else:
  {"tool": "find_patient", "args": {"name": "<full name>"}}
  {"tool": "list_slots", "args": {"doctor": "<doctor name>", "day": "<Mon|Tue|Wed|Thu|Fri>"}}
  {"tool": "book", "args": {"patient_id": "<P...>", "slot_id": "<S...>"}}
  {"tool": "cancel", "args": {"slot_id": "<S...>"}}
  {"done": "<one sentence for the patient>"}
Times are 24-hour HH:MM. list_slots does not sort by time. When two patients share a name, use the date of birth."""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> dict | None:
    """The JSON object in the reply, or None. Strict enough to grade, loose enough for a small model."""
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        action = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return action if isinstance(action, dict) and ("tool" in action or "done" in action) else None


class DeskEnv:
    """reset(task) -> first observation; step(action_text) -> (observation, reward, done, info).

    The reward arrives once, at the end of the episode (a "sparse" reward), as the verifier's visible key; the info
    dict carries every key, so the hidden check can be logged without ever training on it.
    """

    def __init__(self, world: sqlite3.Connection, max_turns: int = 8):
        self.world, self.max_turns = world, max_turns

    def reset(self, task: Task) -> str:
        self.task, self.db, self.turns, self.calls = task, copy_db(self.world), 0, []
        return task.instruction

    def step(self, action_text: str) -> tuple[str, float, bool, dict]:
        self.turns += 1
        action = parse_action(action_text)
        if action is None:
            observation = json.dumps({"error": "reply with one JSON object, as in the system prompt"})
        elif "done" in action:
            return self._finish("done")
        else:
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            result = call_tool(self.db, str(action.get("tool")), args)
            self.calls.append({"tool": action.get("tool"), "args": args, "result": result})
            observation = json.dumps(result)
        if self.turns >= self.max_turns:
            return self._finish("out of turns", observation)
        return observation, 0.0, False, {}

    def _finish(self, why: str, observation: str = "") -> tuple[str, float, bool, dict]:
        scores = verify(self.world, self.db, self.task)
        return observation, scores["reward"], True, {**scores, "end": why, "turns": self.turns}
