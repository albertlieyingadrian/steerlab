"""Step 4 (f): package each task as a Harbor task folder.

    <task>/
      task.toml               schema, timeouts, a separate verifier sandbox, the one artifact that crosses over
      instruction.md          what the agent reads: the request, plus how to use the `desk` command
      environment/Dockerfile  python:3.11-slim + the desk tools + this task's starting calendar (clinic.db)
      tests/Dockerfile        the verifier's own image: it never sees the agent's sandbox, only /app/clinic.db
      tests/test.sh           runs verify.py, which writes /logs/verifier/reward.json with every key, on every path
      solution/solve.sh       the oracle: the tool calls that solve the task (`harbor run -a oracle` must score 1)

The agent's sandbox holds no tests and no expected answer: those live only in the verifier's image, so the
"read the answer key" and "edit the tests" hacks have nothing to read or edit.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from envlab.desk import Task, world_from_json
from envlab.pipeline import oracle_calls

ENVLAB = Path(__file__).resolve().parent
BASE_IMAGE = "python:3.11-slim@sha256:61e01f72680f86a5279ad510818ab7b2ee9f049822589d4e34e33e044360356c"  # 2026-11-24

TASK_TOML = """schema_version = "1.4"
artifacts = ["/app/clinic.db"]

[task]
name = "rl-envs-101/{name}"
description = "Clinic front desk: {kind} an appointment with the desk tools."
authors = []
keywords = ["tool-use", "sqlite", "front-desk"]

[metadata]
difficulty = "easy"
category = "tool-use"
tags = ["{kind}", "from-logs"]

[verifier]
timeout_sec = 60.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 2048
gpus = 0
mcp_servers = []

[agent]
user = "agent"
timeout_sec = 600.0

[environment]
network_mode = "no-network"
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 2048
gpus = 0
mcp_servers = []

[verifier.env]

[environment.env]

[solution.env]
"""

INSTRUCTION = """{request}

You are the clinic's front-desk assistant. Change the calendar only with the `desk` command:

    desk find-patient "<full name>"               patients with that name, their date of birth and appointments
    desk slots "<doctor>" <Mon|Tue|Wed|Thu|Fri>   that doctor's free slots that day
    desk book <patient_id> <slot_id>              book a free slot
    desk cancel <slot_id>                         free a booked slot

Every command prints JSON. Times are 24-hour HH:MM, and `desk slots` does not list them in time order.
When two patients share a name, use the date of birth. Change nothing the request does not ask for.
"""

AGENT_DOCKERFILE = """FROM {base}
# tmux and asciinema: the terminal agent (terminus-2) drives a tmux session; installed at build time, since the
# sandbox itself has no network
RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY desk.py desk_cli.py world.json /opt/desk/
RUN printf '#!/bin/sh\\nexec python3 /opt/desk/desk_cli.py "$@"\\n' > /usr/local/bin/desk && chmod +x /usr/local/bin/desk \\
    && python3 /opt/desk/desk_cli.py --init /app/clinic.db && rm /opt/desk/world.json \\
    && useradd --create-home --shell /bin/bash agent && chown -R agent:agent /app
"""

VERIFIER_DOCKERFILE = """FROM {base}
COPY desk.py verify.py task.json start_world.json /tests/
COPY --chmod=755 test.sh /tests/test.sh
"""

TEST_SH = """#!/bin/bash
# Grades the calendar the agent left, never its words. Writes every key on every path (a missing key averages as 0).
mkdir -p /logs/verifier
python3 /tests/verify.py /app/clinic.db > /logs/verifier/reward.json 2> /logs/verifier/verify_error.txt \\
  || echo '{"reward": 0.0, "hidden": 0.0, "collateral": -1.0}' > /logs/verifier/reward.json
cat /logs/verifier/reward.json
"""

VERIFY_PY = '''"""Harbor verifier: compare the agent's final calendar with the one the task should produce."""

import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from desk import Task, verify, world_from_json

task = Task(**json.loads((HERE / "task.json").read_text()))
start = world_from_json(json.loads((HERE / "start_world.json").read_text()))
final = sqlite3.connect(sys.argv[1])
print(json.dumps(verify(start, final, task)))
'''

DESK_CLI = '''"""The `desk` command inside the agent's sandbox: the same four tools as the Python environment."""

import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from desk import book, cancel, find_patient, list_slots, world_from_json

DB = "/app/clinic.db"
USAGE = "usage: desk find-patient NAME | desk slots DOCTOR DAY | desk book PATIENT_ID SLOT_ID | desk cancel SLOT_ID"
COMMANDS = {"find-patient": (find_patient, 1), "slots": (list_slots, 2), "book": (book, 2), "cancel": (cancel, 1)}


def main(argv, db_path=DB):
    if argv[:1] == ["--init"]:
        out = sqlite3.connect(argv[1])
        world_from_json(json.loads((HERE / "world.json").read_text())).backup(out)
        out.commit()
        return 0
    if not argv or argv[0] not in COMMANDS or len(argv) - 1 != COMMANDS[argv[0]][1]:
        print(json.dumps({"error": USAGE}))
        return 2
    print(json.dumps(COMMANDS[argv[0]][0](sqlite3.connect(db_path), *argv[1:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


def solve_sh(task: Task, world) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "# The oracle, computed from the starting calendar by envlab/pipeline.py",
    ]
    for call in oracle_calls(task, world):
        if call["tool"] == "book":
            lines.append(f"desk book {call['args']['patient_id']} {call['args']['slot_id']}")
        else:
            lines.append(f"desk cancel {call['args']['slot_id']}")
    return "\n".join(lines) + "\n"


def task_name(item: dict) -> str:
    return f"desk-{item['task']['kind']}-{item['task']['task_id']}"


def write_task(item: dict, root: Path) -> Path:
    task = Task(**item["task"])
    folder = root / task_name(item)
    if folder.exists():
        shutil.rmtree(folder)
    for sub in ("environment", "tests", "solution"):
        (folder / sub).mkdir(parents=True)
    files = {
        "task.toml": TASK_TOML.format(name=task_name(item), kind=task.kind),
        "instruction.md": INSTRUCTION.format(request=task.instruction),
        "environment/Dockerfile": AGENT_DOCKERFILE.format(base=BASE_IMAGE),
        "environment/desk_cli.py": DESK_CLI,
        "environment/world.json": json.dumps(item["world"]),
        "tests/Dockerfile": VERIFIER_DOCKERFILE.format(base=BASE_IMAGE),
        "tests/test.sh": TEST_SH,
        "tests/verify.py": VERIFY_PY,
        "tests/task.json": json.dumps(item["task"]),
        "tests/start_world.json": json.dumps(item["world"]),
        "solution/solve.sh": solve_sh(task, world_from_json(item["world"])),
    }
    for path, text in files.items():
        (folder / path).write_text(text)
    for sub in ("environment", "tests"):
        shutil.copy(ENVLAB / "desk.py", folder / sub / "desk.py")
    for script in ("tests/test.sh", "solution/solve.sh"):
        (folder / script).chmod(0o755)
    return folder
