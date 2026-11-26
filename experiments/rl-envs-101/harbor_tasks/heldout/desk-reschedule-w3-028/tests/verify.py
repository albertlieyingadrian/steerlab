"""Harbor verifier: compare the agent's final calendar with the one the task should produce."""

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
