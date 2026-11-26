"""Step 4, steps (a)-(e): turn the logs into train and held-out tasks, then grade each step against the truth file.

Reads data/logs/desk_logs.jsonl (what a real system keeps). Writes data/tasks/{train,heldout}.jsonl and
results/04_pipeline.json. The truth file (data/logs/truth.jsonl) is read only after each step, to say how well that
step did; nothing the pipeline decides depends on it.

    python experiments/rl-envs-101/04b_pipeline.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envlab import pipeline as P
from envlab.desk import DAYS, DOCTORS, call_tool, copy_db, verify, world_from_json, world_to_json
from envlab.logs import dump_jsonl, load_jsonl

LOGS = Path(os.environ.get("RLENV_LOGS", HERE / "data" / "logs"))  # overridable for the tests
OUT = Path(os.environ.get("RLENV_OUT", HERE / "data" / "tasks"))
RESULTS = Path(os.environ.get("RLENV_RESULTS", HERE / "results"))
SPLIT_DATE = "2026-03-16"  # weeks 1-2 train, week 3 held out
report: dict = {}
checks: list[bool] = []


def check(ok: bool, text: str) -> None:
    checks.append(bool(ok))
    print(("✅" if ok else "❌"), text)


def rate(pair) -> str:
    return f"{pair[0]:3d}/{pair[1]:3d} = {pair[0] / max(1, pair[1]):.2f}"


logs = load_jsonl(LOGS / "desk_logs.jsonl")
truth = {t["session_id"]: t for t in load_jsonl(LOGS / "truth.jsonl")}
sessions = [P.parse_session(log) for log in logs]
feedback = dict(Counter(s["feedback"] for s in sessions))
print(f"{len(sessions)} logged sessions from {len({s['agent'] for s in sessions})} agents; feedback {feedback}")

# (a) cluster ---------------------------------------------------------------------------------------------------------
labels, info = P.cluster(sessions)
names = P.name_clusters(sessions, labels)
kinds = [names[c] for c in labels]
print(f"\n(a) cluster: silhouette by k {info['silhouette']}, chose k = {info['k']}, named {names}")
confusion = Counter((truth[s["session_id"]]["task"]["kind"], k) for s, k in zip(sessions, kinds))
agreement = sum(n for (t, k), n in confusion.items() if t == k) / len(sessions)
print("    true type -> cluster name:", dict(sorted(confusion.items())))
print(f"    agreement with the true type: {agreement:.3f}")
report["cluster"] = {**info, "names": {str(k): v for k, v in names.items()}, "agreement": round(agreement, 4)}
report["cluster"]["confusion"] = {f"{t}->{k}": n for (t, k), n in sorted(confusion.items())}
check(agreement > 0.9, "clusters recover the three task types for over 90% of sessions")

# (b) reconstruct -----------------------------------------------------------------------------------------------------
listings = [
    c["result"] for s in sessions for c in s["calls"] if c["tool"] == "list_slots" and "free" in (c["result"] or {})
]
doctor_ids = {name: f"D{i + 1}" for i, name in enumerate(sorted({r["doctor"] for r in listings}))}
times = sorted({x["time"] for r in listings for x in r["free"]})
days = [d for d in DAYS if any(r["day"] == d for r in listings)]
print(
    f"\n(b) reconstruct: {len(doctor_ids)} doctors, hours {times[0]}-{times[-1]} ({len(times)} slots a day), days {days}"
)
rebuilt = []
for s in sessions:
    obs = P.observed_start(s)
    world = P.build_world(obs, doctor_ids, times, days)
    rebuilt.append((obs, world, P.replay_matches(s, world)))
fidelity = sum(r[2] for r in rebuilt) / len(rebuilt)
print(f"    replay fidelity: {fidelity:.3f} of sessions reproduce every logged tool result on their rebuilt world")
report["reconstruct"] = {"doctors": sorted(doctor_ids), "times": times, "replay_fidelity": round(fidelity, 4)}
check(fidelity > 0.95, "rebuilt worlds reproduce the logged tool results in over 95% of sessions")

# (c) extract ---------------------------------------------------------------------------------------------------------
extracted, reasons = [], Counter()
for s, kind, (obs, world, ok) in zip(sessions, kinds, rebuilt):
    if not ok or kind == "unknown":
        reasons["replay mismatch or unknown type"] += 1
        continue
    task, why = P.extract_task(s, kind, obs, doctor_ids)
    reasons[why] += 1
    if task:
        extracted.append((s, task, world))
templates = Counter((t.kind, P.normalise(s["instruction"], P.patient_names(s))) for s, t, _ in extracted)
print(f"\n(c) extract: {len(extracted)} tasks; outcomes {dict(reasons)}")
print(f"    {len(templates)} prompt templates (the seed prompts); the most common:")
for (kind, template), n in templates.most_common(4):
    print(f"      [{kind}] x{n}: {template}")
doctor_of = {v: k for k, v in doctor_ids.items()}
true_doctor = dict(DOCTORS)
exact = 0
for s, task, _world in extracted:
    t = truth[s["session_id"]]["task"]
    got = (task.kind, task.patient_id, doctor_of[task.doctor_id], task.day, task.old_slot_id)
    exact += got == (t["kind"], t["patient_id"], true_doctor[t["doctor_id"]], t["day"], t["old_slot_id"])
print(f"    extracted task equals the true task: {exact}/{len(extracted)}")
report["extract"] = {"tasks": len(extracted), "outcomes": dict(reasons), "templates": len(templates), "exact": exact}
check(exact / max(1, len(extracted)) > 0.95, "extracted tasks match the true task in over 95% of cases")

# (d) derive the verifier ---------------------------------------------------------------------------------------------
evidence = P.slot_rule_evidence(sessions, kinds)
rule = P.choose_rule(evidence)
print("\n(d) verifier: which slot did the sessions people were happy with book?")
for r, ev in evidence.items():
    print(f"    {r:13s} resolved {rate(ev['resolved'])}   not resolved {rate(ev['not_resolved'])}")
writes = P.writes_per_resolved_session(sessions, kinds)
print(f"    chosen slot rule: {rule}; successful writes per resolved session, by type: {writes}")
report["verifier"] = {"evidence": evidence, "rule": rule}
report["verifier"]["writes_resolved"] = {k: {str(a): b for a, b in v.items()} for k, v in writes.items()}
check(rule == "earliest", "the derived rule is 'earliest free slot' (what the clinic asked for)")

agree_truth = agree_feedback = n_feedback = 0
for s, task, world in extracted:
    end = copy_db(world)
    for c in s["calls"]:
        if c["result"] is not None:
            call_tool(end, str(c["tool"]), c["args"])
    derived = verify(world, end, task)["hidden"]
    agree_truth += derived == truth[s["session_id"]]["scores"]["hidden"]
    if s["feedback"]:
        n_feedback += 1
        agree_feedback += derived == (s["feedback"] == "resolved")
print(
    f"    derived verifier vs the true hidden check {agree_truth}/{len(extracted)}; vs patient feedback {agree_feedback}/{n_feedback}"
)
report["verifier"].update(
    {"agree_truth": [agree_truth, len(extracted)], "agree_feedback": [agree_feedback, n_feedback]}
)
check(agree_truth == len(extracted), "on extracted tasks the derived verifier agrees with the true check every time")

# quality checks, then (e) dedupe and split ---------------------------------------------------------------------------
raw_private = sum(bool(P.private_data(s["instruction"])) for s in sessions)
packed, drops, seen = [], Counter(), set()
for s, task, world in extracted:
    oracle = copy_db(world)
    for c in P.oracle_calls(task, world):
        call_tool(oracle, c["tool"], c["args"])
    scores = verify(world, oracle, task)
    if scores["hidden"] != 1.0 or scores["reward"] != 1.0:
        drops["oracle fails"] += 1
        continue
    if verify(world, copy_db(world), task)["hidden"] != 0.0:
        drops["doing nothing passes"] += 1
        continue
    missing = P.instruction_complete(task, world)
    if missing:
        drops["instruction incomplete: " + ", ".join(missing)] += 1
        continue
    if P.private_data(task.instruction):
        drops["private data left"] += 1
        continue
    key = P.task_key(task, world)
    if key in seen:
        drops["duplicate"] += 1
        continue
    seen.add(key)
    meta = {
        "template": P.normalise(s["instruction"], P.patient_names(s)),
        "agent": s["agent"],
        "feedback": s["feedback"],
        "started_at": s["started_at"],
    }
    split = "train" if s["started_at"] < SPLIT_DATE else "heldout"
    packed.append(
        {"id": task.task_id, "split": split, "task": task.to_json(), "world": world_to_json(world), "meta": meta}
    )
print(f"\nQuality checks: {raw_private} raw requests held a phone number (all redacted); dropped {dict(drops)}")
train = [p for p in packed if p["split"] == "train"]
held = [p for p in packed if p["split"] == "heldout"]
kinds_train = dict(Counter(p["task"]["kind"] for p in train))
kinds_held = dict(Counter(p["task"]["kind"] for p in held))
print(f"(e) split by date: {len(train)} train {kinds_train}, {len(held)} held-out {kinds_held}")


def request(p: dict) -> tuple:
    w = world_from_json(p["world"])
    name, dob = w.execute("SELECT name, dob FROM patients WHERE id=?", (p["task"]["patient_id"],)).fetchone()
    return (p["task"]["kind"], name, dob, p["task"]["doctor_id"], p["task"]["day"])


train_requests = {request(p) for p in train}
train_templates = {p["meta"]["template"] for p in train}
train_texts = {p["task"]["instruction"] for p in train}
same_request = sum(request(p) in train_requests for p in held)
same_template = sum(p["meta"]["template"] in train_templates for p in held)
same_text = sum(p["task"]["instruction"] in train_texts for p in held)
print(
    f"    contamination: held-out tasks whose request (type, patient, doctor, day) is also in train {same_request}/{len(held)};"
)
print(
    f"    whose wording template is in train {same_template}/{len(held)}; whose exact text is in train {same_text}/{len(held)}"
)
report["quality"] = {"raw_with_phone": raw_private, "drops": dict(drops)}
report["split"] = {
    "train": len(train),
    "heldout": len(held),
    "kinds_train": kinds_train,
    "kinds_heldout": kinds_held,
    "heldout_same_request_in_train": same_request,
    "heldout_same_template_in_train": same_template,
    "heldout_same_text_in_train": same_text,
}
check(same_text <= 0.05 * len(held), "at most 5% of held-out request texts appear verbatim in train")
check(all(p["meta"]["started_at"] >= SPLIT_DATE for p in held), "every held-out task comes after every train task")
dump_jsonl(train, OUT / "train.jsonl")
dump_jsonl(held, OUT / "heldout.jsonl")
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "04_pipeline.json").write_text(json.dumps(report, indent=1))
print(f"\nwrote data/tasks/train.jsonl and data/tasks/heldout.jsonl\n{sum(checks)}/{len(checks)} checks passed")
