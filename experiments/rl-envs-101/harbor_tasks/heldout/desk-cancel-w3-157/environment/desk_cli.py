"""The `desk` command inside the agent's sandbox: the same four tools as the Python environment."""

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
