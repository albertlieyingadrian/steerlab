"""Step 4: from logged trajectories to an environment, in six steps.

The input is only what a real system keeps (envlab/logs.py): chats with tool calls and results, a timestamp, an
agent name and the patient's feedback. The output is a set of tasks, each with a starting world, an instruction, a
verifier and an oracle solution, split into train and held-out.

    (a) cluster      group sessions into task types from the request's words and the tools used
    (b) reconstruct  rebuild each session's starting world from what its tool results revealed
    (c) extract      pull out the task (who, which doctor, which day) and a reusable prompt template
    (d) verifiers    find the rule the resolved sessions followed ("earliest slot"), and turn it into a check
    (e) split        drop duplicates, split by date (train = first two weeks), and measure the overlap
    (f) package      Harbor task folders (envlab/harbor_pack.py)

An analogy: the logs are CCTV footage of a front desk. From footage alone you work out what kinds of requests come
in (a), what the calendar looked like when each started (b), what each patient asked for (c), what "done right"
looked like in the sessions people were happy with (d), and you keep last week's footage aside for the exam (e).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict

from envlab.desk import SCHEMA, Task, call_tool, copy_db, earliest_free, parse_action, slot_table

PHONE_RE = re.compile(r"\b\d{3}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
DAY_RE = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri)(?:day|sday|nesday|rsday|iday)?\b")
DOCTOR_RE = re.compile(r"\bDr\.? ([A-Z][a-z]+)")
DOB_RE = re.compile(r"\(born (\d{4}-\d{2}-\d{2})\)")
PLACEHOLDER = ("P999", "Booked Elsewhere", "1900-01-01", "")

# --- parsing ---------------------------------------------------------------------------------------------------------


def parse_session(log: dict) -> dict:
    """Turn a chat into (instruction, tool calls with their results, final reply)."""
    messages = log["messages"]
    calls, final = [], None
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        action = parse_action(m["content"])
        if action is None:
            continue
        if "done" in action:
            final = action["done"]
            break
        result = None
        if i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
            try:
                result = json.loads(messages[i + 1]["content"])
            except json.JSONDecodeError:
                result = None
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        calls.append({"tool": action.get("tool"), "args": args, "result": result})
    return {**log, "instruction": messages[0]["content"], "calls": calls, "final": final}


def patient_names(session: dict) -> list[str]:
    names = set()
    for c in session["calls"]:
        if c["tool"] == "find_patient":
            names.add(str(c["args"].get("name", "")))
            names.update(m["name"] for m in (c["result"] or {}).get("matches", []))
    return [n for n in names if n and n in session["instruction"]]


def normalise(text: str, names: list[str]) -> str:
    """Replace the parts that change between requests with placeholders, keeping the wording: a prompt template."""
    text = PHONE_RE.sub("{phone}", text)
    text = DOB_RE.sub("{dob}", text)
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(name, "{patient}")
    text = DOCTOR_RE.sub("{doctor}", text)
    return DAY_RE.sub("{day}", text)


# --- (a) cluster -----------------------------------------------------------------------------------------------------


def tool_signature(session: dict) -> str:
    """Which writing tools succeeded, in order of first use: the session's 'shape'."""
    writes = [c["tool"] for c in session["calls"] if c["tool"] in ("book", "cancel") and (c["result"] or {}).get("ok")]
    return "+".join(dict.fromkeys(writes)) or "none"


def cluster(sessions: list[dict], k_range=range(2, 13), seed: int = 0) -> tuple[list[int], dict]:
    """KMeans on TF-IDF of the request template plus its tool signature; k chosen by silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    docs = [
        normalise(s["instruction"], patient_names(s)) + " sig_" + tool_signature(s).replace("+", "_") for s in sessions
    ]
    features = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(docs)
    scores, fits = {}, {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(features)
        scores[k] = round(float(silhouette_score(features, labels, random_state=seed)), 4)
        fits[k] = [int(x) for x in labels]
    best = max(scores, key=scores.get)
    return fits[best], {"silhouette": scores, "k": best}


def name_clusters(sessions: list[dict], labels: list[int]) -> dict[int, str]:
    """Name each cluster after the write pattern of its resolved sessions (book, book+cancel, cancel)."""
    shape_to_kind = {"book": "book", "book+cancel": "reschedule", "cancel+book": "reschedule", "cancel": "cancel"}
    names = {}
    for c in sorted(set(labels)):
        shapes = Counter(
            tool_signature(s) for s, label in zip(sessions, labels) if label == c and s["feedback"] == "resolved"
        )
        top = next((shape for shape, _ in shapes.most_common() if shape in shape_to_kind), "none")
        names[c] = shape_to_kind.get(top, "unknown")
    return names


# --- (b) reconstruct the starting world -------------------------------------------------------------------------------


def observed_start(session: dict) -> dict:
    """What the session's tool results say the world looked like before its own writes.

    A read made after one of the session's own writes is undone first: a slot it booked was free at the start, and
    a slot it cancelled was booked at the start.
    """
    patients: dict = {}
    booked: dict = {}
    free: dict = {}
    own_booked: dict = {}
    own_cancelled: set = set()
    for c in session["calls"]:
        r = c["result"] or {}
        if c["tool"] == "find_patient":
            for m in r.get("matches", []):
                patients.setdefault(m["patient_id"], (m["name"], m["dob"]))
                for a in m["appointments"]:
                    if a["slot_id"] not in own_booked:
                        booked.setdefault(a["slot_id"], (a["doctor"], a["day"], a["time"], m["patient_id"]))
        elif c["tool"] == "list_slots" and "free" in r:
            key = (r["doctor"], r["day"])
            if key not in free:
                listed = {s["slot_id"]: s["time"] for s in r["free"] if s["slot_id"] not in own_cancelled}
                for sid, where in own_booked.items():
                    if where[:2] == key:
                        listed[sid] = where[2]
                free[key] = listed
        elif c["tool"] == "book" and r.get("ok"):
            own_booked[r["slot_id"]] = (r["doctor"], r["day"], r["time"])
        elif c["tool"] == "cancel" and r.get("ok"):
            own_cancelled.add(c["args"].get("slot_id"))
    return {"patients": patients, "booked": booked, "free": free, "own_booked": own_booked}


def build_world(obs: dict, doctor_ids: dict[str, str], times: list[str], days: list[str]) -> sqlite3.Connection:
    """Pin every observed fact; fill every unobserved slot as booked by a placeholder, so it can't change an answer."""
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    for pid, (name, dob) in sorted(obs["patients"].items()):
        db.execute("INSERT INTO patients VALUES (?,?,?,?)", (pid, name, dob, ""))  # phone dropped: no task needs it
    db.execute("INSERT INTO patients VALUES (?,?,?,?)", PLACEHOLDER)
    db.executemany("INSERT INTO doctors VALUES (?,?)", sorted((did, name) for name, did in doctor_ids.items()))
    taken: dict[tuple, str] = {}
    rows = []

    def add(sid, doctor, day, time, pid):
        if (doctor, day, time) not in taken and sid not in taken.values():
            rows.append((sid, doctor_ids[doctor], day, time, pid))
            taken[(doctor, day, time)] = sid

    for sid, (doctor, day, time, pid) in obs["booked"].items():
        add(sid, doctor, day, time, pid)
    for (doctor, day), listed in obs["free"].items():
        for sid, time in listed.items():
            add(sid, doctor, day, time, None)
    for sid, (doctor, day, time) in obs["own_booked"].items():
        add(sid, doctor, day, time, None)  # the session booked it, so it was free at the start
    n = 0
    for doctor in doctor_ids:
        for day in days:
            for time in times:
                if (doctor, day, time) not in taken:
                    n += 1
                    add(f"X{n:03d}", doctor, day, time, PLACEHOLDER[0])
    db.executemany("INSERT INTO slots VALUES (?,?,?,?,?)", rows)
    db.commit()
    return db


def strip_phone(value):
    if isinstance(value, dict):
        return {k: strip_phone(v) for k, v in value.items() if k != "phone"}
    if isinstance(value, list):
        return [strip_phone(v) for v in value]
    return value


def replay_matches(session: dict, world: sqlite3.Connection) -> bool:
    """Replay the session's tool calls on the rebuilt world: every result must equal the logged one (phones aside)."""
    db = copy_db(world)
    for c in session["calls"]:
        if c["result"] is None:
            continue
        got = call_tool(db, str(c["tool"]), c["args"])
        if strip_phone(got) != strip_phone(c["result"]):
            return False
    return True


# --- (c) extract the task -------------------------------------------------------------------------------------------


def redact(text: str) -> str:
    """Remove contact details from a request (no task needs them)."""
    text = re.sub(r"\s*\(\s*\d{3}-\d{4}\s*\)", "", text)
    text = re.sub(r"\s*(You can reach me at|Call) \d{3}-\d{4}( if needed)?\.", "", text)
    return EMAIL_RE.sub("[email removed]", PHONE_RE.sub("[phone removed]", text))


def extract_task(session: dict, kind: str, obs: dict, doctor_ids: dict[str, str]) -> tuple[Task | None, str]:
    """Read who / which doctor / which day from the request, checked against what the tools returned."""
    text = session["instruction"]
    found = DOCTOR_RE.search(text)
    doctor = f"Dr {found.group(1)}" if found else None
    if doctor not in doctor_ids:
        return None, "no known doctor in the request"
    day_match = DAY_RE.search(text)
    day = day_match.group(1) if day_match else None
    if kind in ("book", "reschedule") and day is None:
        return None, "no day in the request"
    names = patient_names(session)
    if not names:
        return None, "patient never looked up"
    name = max(names, key=len)
    matches = [pid for pid, (n, _) in obs["patients"].items() if n == name]
    dob = DOB_RE.search(text)
    if dob:
        matches = [pid for pid in matches if obs["patients"][pid][1] == dob.group(1)]
    if len(matches) != 1:
        return None, "patient ambiguous"
    pid = matches[0]
    old = None
    if kind in ("reschedule", "cancel"):
        olds = [sid for sid, (d, _, _, p) in obs["booked"].items() if p == pid and d == doctor]
        if not olds:
            return None, "the appointment to change was never seen"
        if len(olds) > 1:
            return None, "two appointments with that doctor (the request is ambiguous)"
        old = olds[0]
    if kind in ("book", "reschedule") and (doctor, day) not in obs["free"]:
        return None, "the target day's free slots were never listed"
    if kind == "book" and any(p == pid and d == doctor for d, _, _, p in obs["booked"].values()):
        return None, "patient already booked with that doctor"
    target_day = day if kind != "cancel" else None
    return Task(session["session_id"], kind, redact(text), pid, doctor_ids[doctor], target_day, old), "ok"


# --- (d) derive the verifier ----------------------------------------------------------------------------------------

SLOT_RULES = {
    "earliest": lambda free: min(free, key=free.get),
    "latest": lambda free: max(free, key=free.get),
    "first listed": lambda free: next(iter(free)),
}


def slot_rule_evidence(sessions: list[dict], kinds: list[str]) -> dict:
    """For each candidate rule: how often the booked slot is the one the rule picks, among resolved and unresolved."""
    counts = {rule: {"resolved": [0, 0], "not_resolved": [0, 0]} for rule in SLOT_RULES}
    for s, kind in zip(sessions, kinds):
        if kind not in ("book", "reschedule") or s["feedback"] not in ("resolved", "not_resolved"):
            continue
        lists = [c for c in s["calls"] if c["tool"] == "list_slots" and (c["result"] or {}).get("free")]
        books = [c for c in s["calls"] if c["tool"] == "book" and (c["result"] or {}).get("ok")]
        if not lists or len(books) != 1:
            continue
        free = {x["slot_id"]: x["time"] for x in lists[0]["result"]["free"]}
        chosen = books[0]["args"].get("slot_id")
        for rule, pick in SLOT_RULES.items():
            counts[rule][s["feedback"]][0] += int(pick(free) == chosen)
            counts[rule][s["feedback"]][1] += 1
    return counts


def choose_rule(evidence: dict) -> str:
    """The rule that best separates happy from unhappy sessions: P(rule | resolved) - P(rule | not resolved)."""

    def rate(pair):
        return pair[0] / pair[1] if pair[1] else 0.0

    return max(evidence, key=lambda r: rate(evidence[r]["resolved"]) - rate(evidence[r]["not_resolved"]))


def writes_per_resolved_session(sessions: list[dict], kinds: list[str]) -> dict[str, dict[int, int]]:
    """How many successful writes the resolved sessions of each kind made: evidence for 'touch nothing else'."""
    out: dict[str, Counter] = defaultdict(Counter)
    for s, kind in zip(sessions, kinds):
        if s["feedback"] == "resolved":
            out[kind][
                sum(c["tool"] in ("book", "cancel") and bool((c["result"] or {}).get("ok")) for c in s["calls"])
            ] += 1
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def oracle_calls(task: Task, world: sqlite3.Connection) -> list[dict]:
    """The tool calls that solve a task, computed from its starting world (Harbor's solution/solve.sh)."""
    calls = []
    if task.kind in ("book", "reschedule"):
        slot = earliest_free(world, task.doctor_id, task.day)
        calls.append({"tool": "book", "args": {"patient_id": task.patient_id, "slot_id": slot}})
    if task.old_slot_id:
        calls.append({"tool": "cancel", "args": {"slot_id": task.old_slot_id}})
    return calls


# --- quality checks (the kind a shared-dataset tool runs on every task) ----------------------------------------------

EARLY_WORDS = ("earliest", "first opening", "as early as possible")


def instruction_complete(task: Task, world: sqlite3.Connection) -> list[str]:
    """Everything the hidden check grades must be stated in the instruction. Returns what is missing."""
    missing = []
    name, dob = world.execute("SELECT name, dob FROM patients WHERE id=?", (task.patient_id,)).fetchone()
    doctor = world.execute("SELECT name FROM doctors WHERE id=?", (task.doctor_id,)).fetchone()[0]
    text = task.instruction
    if name not in text:
        missing.append("patient name")
    if world.execute("SELECT count(*) FROM patients WHERE name=?", (name,)).fetchone()[0] > 1 and dob not in text:
        missing.append("date of birth (the name is shared)")
    if doctor not in text:
        missing.append("doctor")
    if task.kind in ("book", "reschedule"):
        if not DAY_RE.search(text):
            missing.append("day")
        if not any(w in text.lower() for w in EARLY_WORDS):
            missing.append("which slot (earliest)")
    return missing


def private_data(text: str) -> list[str]:
    return PHONE_RE.findall(text) + EMAIL_RE.findall(text)


def task_key(task: Task, world: sqlite3.Connection) -> str:
    """Two tasks are duplicates when they ask the same thing of the same starting calendar."""
    name, dob = world.execute("SELECT name, dob FROM patients WHERE id=?", (task.patient_id,)).fetchone()
    return json.dumps(
        [task.kind, name, dob, task.doctor_id, task.day, task.old_slot_id, sorted(slot_table(world).items())]
    )
