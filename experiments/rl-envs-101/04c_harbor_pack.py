"""Step 4 (f): write Harbor task folders for 30 held-out tasks, then run the dataset checks on each, locally.

    python experiments/rl-envs-101/04c_harbor_pack.py

Checks, the kind a shared RL-environment workspace runs on every task change:
    oracle-passes          solution/solve.sh, run against the task's own desk tools, scores reward = hidden = 1
    nop-fails              leaving the calendar alone scores hidden = 0
    instruction-complete   everything the hidden check grades is stated in instruction.md
    no-private-data        no phone number or email in any file of the task folder
    no-answer-in-sandbox   the agent's environment/ folder holds no expected answer or test file

Writes harbor_tasks/heldout/<task>/ and results/04_harbor_checks.json. No Docker needed: the commands run with the
same Python files the containers use. `harbor run -a oracle` in step 5 repeats the oracle check in real sandboxes.
"""

import contextlib
import io
import json
import random
import runpy
import shlex
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envlab import harbor_pack, pipeline
from envlab.desk import Task, world_from_json
from envlab.logs import load_jsonl

ROOT = HERE / "harbor_tasks" / "heldout"
N_PER_KIND = 10
CHECKS = ("oracle_passes", "nop_fails", "instruction_complete", "no_private_data", "no_answer_in_sandbox")


def run_in_sandbox(folder: Path, commands: list[str]) -> dict:
    """Build the task's clinic.db, run `desk` commands against it, then run the verifier on the result."""
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / "env"
        shutil.copytree(folder / "environment", env)
        db = Path(tmp) / "clinic.db"
        cli = runpy.run_path(str(env / "desk_cli.py"))
        cli["main"](["--init", str(db)])
        for line in commands:
            words = shlex.split(line)
            if words and words[0] == "desk":
                with contextlib.redirect_stdout(io.StringIO()):
                    cli["main"](words[1:], str(db))
        tests = Path(tmp) / "tests"
        shutil.copytree(folder / "tests", tests)
        out, saved = io.StringIO(), sys.argv
        sys.argv = [str(tests / "verify.py"), str(db)]
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(str(tests / "verify.py"), run_name="__main__")
        finally:
            sys.argv = saved
            sys.modules.pop("desk", None)
        return json.loads(out.getvalue())


held = load_jsonl(HERE / "data" / "tasks" / "heldout.jsonl")
rng = random.Random(0)
chosen = []
for kind in ("book", "reschedule", "cancel"):
    pool = sorted((p for p in held if p["task"]["kind"] == kind), key=lambda p: p["id"])
    chosen += rng.sample(pool, min(N_PER_KIND, len(pool)))
if ROOT.exists():
    shutil.rmtree(ROOT)
results, failures = [], Counter()
for item in chosen:
    folder = harbor_pack.write_task(item, ROOT)
    oracle = run_in_sandbox(folder, (folder / "solution" / "solve.sh").read_text().splitlines())
    nop = run_in_sandbox(folder, [])
    missing = pipeline.instruction_complete(Task(**item["task"]), world_from_json(item["world"]))
    files = [p for p in folder.rglob("*") if p.is_file()]
    private = [str(p.relative_to(folder)) for p in files if pipeline.private_data(p.read_text())]
    leaked = [p.name for p in (folder / "environment").iterdir() if p.name in ("task.json", "verify.py", "test.sh")]
    row = {"task": folder.name, "oracle": oracle, "nop": nop, "missing": missing, "private": private}
    row.update(
        {
            "oracle_passes": oracle["reward"] == oracle["hidden"] == 1.0,
            "nop_fails": nop["hidden"] == 0.0,
            "instruction_complete": not missing,
            "no_private_data": not private,
            "no_answer_in_sandbox": not leaked,
        }
    )
    for key in CHECKS:
        failures[key] += not row[key]
    results.append(row)

print(
    f"wrote {len(results)} Harbor tasks to harbor_tasks/heldout/ ({dict(Counter(p['task']['kind'] for p in chosen))})"
)
example = ROOT / results[0]["task"]
print(f"\nexample: {example.name}")
for rel in ("instruction.md", "solution/solve.sh", "tests/test.sh"):
    print(f"--- {rel}\n{(example / rel).read_text().strip()}")
print(f"\nnop's visible reward: {sum(r['nop']['reward'] for r in results):.0f}/{len(results)}")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "04_harbor_checks.json").write_text(
    json.dumps({"tasks": results, "failures": dict(failures)}, indent=1)
)

ok = []
for key in CHECKS:
    ok.append(failures[key] == 0)
    print(("✅" if ok[-1] else "❌"), f"{key}: {len(results) - failures[key]}/{len(results)}")
toml = (example / "task.toml").read_text()
ok.append('environment_mode = "separate"' in toml and 'network_mode = "no-network"' in toml)
print(("✅" if ok[-1] else "❌"), "the verifier runs in its own sandbox, with no network on either side")
print(f"{sum(ok)}/{len(ok)} checks passed")
